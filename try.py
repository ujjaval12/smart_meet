import cv2
import numpy as np
import os
import pickle
import time
import threading

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
ENROLLMENT_FILE      = "enrolled_face.pkl"
ENROLL_SAMPLES       = 20
MATCH_TOLERANCE      = 0.80
MATCH_MARGIN         = 0.06
STABLE_MATCH_FRAMES  = 6
PROTECTED_HOLD_FRAMES = 8
BLUR_PADDING         = 0.3
PROCESS_EVERY_N      = 3       # Reduced from 4 — GPU is fast enough
MIN_DETECT_PROB      = 0.95    # Enrollment: high confidence only
RUN_DETECT_PROB      = 0.80    # Runtime: confident detections only
BLUR_KERNEL          = (51, 51)


# ============================================================
# KALMAN FILTER TRACKER
# Predicts where a face box will be BEFORE recognition catches up.
# This eliminates the lag where blur box trails behind a moving person.
#
# State vector: [cx, cy, w, h, vx, vy, vw, vh]
#   cx, cy = centre x/y   w, h = box width/height
#   vx, vy = velocity x/y  vw, vh = size change velocity
# ============================================================
class KalmanBoxTracker:
    count = 0

    def __init__(self, box):
        # box format: (top, right, bottom, left)
        self.kf = cv2.KalmanFilter(8, 4)

        # Transition matrix — constant velocity model
        self.kf.transitionMatrix = np.array([
            [1,0,0,0, 1,0,0,0],
            [0,1,0,0, 0,1,0,0],
            [0,0,1,0, 0,0,1,0],
            [0,0,0,1, 0,0,0,1],
            [0,0,0,0, 1,0,0,0],
            [0,0,0,0, 0,1,0,0],
            [0,0,0,0, 0,0,1,0],
            [0,0,0,0, 0,0,0,1],
        ], dtype=np.float32)

        # Measurement matrix — we observe [cx, cy, w, h]
        self.kf.measurementMatrix = np.array([
            [1,0,0,0, 0,0,0,0],
            [0,1,0,0, 0,0,0,0],
            [0,0,1,0, 0,0,0,0],
            [0,0,0,1, 0,0,0,0],
        ], dtype=np.float32)

        self.kf.processNoiseCov      = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov  = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost         = np.eye(8, dtype=np.float32)

        cx, cy, w, h = self._box_to_cxywh(box)
        self.kf.statePost = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32)

        self.id           = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.label        = None   # "protected" or "stranger"
        self.missed       = 0      # frames since last detection match
        self.MAX_MISSED   = 10     # drop tracker after this many missed frames

    def predict(self):
        self.kf.predict()
        self.missed += 1

    def update(self, box):
        cx, cy, w, h = self._box_to_cxywh(box)
        measurement = np.array([[cx],[cy],[w],[h]], dtype=np.float32)
        self.kf.correct(measurement)
        self.missed = 0

    def get_box(self):
        s = self.kf.statePost
        cx, cy, w, h = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        top    = int(cy - h / 2)
        bottom = int(cy + h / 2)
        left   = int(cx - w / 2)
        right  = int(cx + w / 2)
        return (top, right, bottom, left)

    def is_dead(self):
        return self.missed > self.MAX_MISSED

    @staticmethod
    def _box_to_cxywh(box):
        top, right, bottom, left = box
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        w  = float(right - left)
        h  = float(bottom - top)
        return cx, cy, w, h


def iou(boxA, boxB):
    """Intersection over Union between two (top,right,bottom,left) boxes."""
    tA, rA, bA, lA = boxA
    tB, rB, bB, lB = boxB
    inter_w = max(0, min(rA, rB) - max(lA, lB))
    inter_h = max(0, min(bA, bB) - max(tA, tB))
    inter   = inter_w * inter_h
    union   = (rA-lA)*(bA-tA) + (rB-lB)*(bB-tB) - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# MULTI-OBJECT KALMAN TRACKER
# Manages one KalmanBoxTracker per detected face.
# Matches new detections to existing trackers via IoU,
# predicts position every frame (no lag), and drops
# trackers that haven't been seen for MAX_MISSED frames.
# ============================================================
class KalmanTracker:
    IOU_THRESHOLD = 0.25   # minimum overlap to consider same face

    def __init__(self):
        self.trackers = []   # list of KalmanBoxTracker

    def update(self, detected_protected, detected_strangers):
        """
        Call with fresh recognition results (can be empty lists on non-recognition frames).
        Returns (protected_boxes, stranger_boxes) with Kalman-predicted positions.
        """
        # Step 1: predict all existing trackers forward one frame
        for t in self.trackers:
            t.predict()

        # Combine detections with labels
        detections = [(b, "protected") for b in detected_protected] + \
                     [(b, "stranger")  for b in detected_strangers]

        if detections:
            # Step 2: match detections to trackers via IoU
            unmatched_dets = list(range(len(detections)))

            for tracker in self.trackers:
                best_iou   = self.IOU_THRESHOLD
                best_det   = -1
                pred_box   = tracker.get_box()

                for di in unmatched_dets:
                    score = iou(pred_box, detections[di][0])
                    if score > best_iou:
                        best_iou = score
                        best_det = di

                if best_det >= 0:
                    det_box, det_label = detections[best_det]
                    tracker.update(det_box)
                    tracker.label  = det_label
                    tracker.missed = 0
                    unmatched_dets.remove(best_det)

            # Step 3: spawn new trackers for unmatched detections
            for di in unmatched_dets:
                det_box, det_label = detections[di]
                t = KalmanBoxTracker(det_box)
                t.label = det_label
                self.trackers.append(t)

        # Step 4: remove dead trackers
        self.trackers = [t for t in self.trackers if not t.is_dead()]

        # Step 5: return predicted box positions
        protected, strangers = [], []
        for t in self.trackers:
            if t.label == "protected":
                protected.append(t.get_box())
            elif t.label == "stranger":
                strangers.append(t.get_box())

        return protected, strangers


# ============================================================
# THREADED CAMERA — non-blocking cap.read()
# ============================================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret     = False
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.thread  = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret   = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


# ============================================================
# GPU SETUP
# ============================================================
def ensure_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA GPU is required. No GPU detected.")


def load_models(device):
    mtcnn  = MTCNN(keep_all=True, device=device,
                   thresholds=[0.5, 0.6, 0.6], min_face_size=20)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ============================================================
# FACE DETECTION + EMBEDDING (GPU)
# ============================================================
def get_faces_and_embeddings(bgr, mtcnn, resnet, device):
    img = bgr_to_pil(bgr)
    boxes, probs = mtcnn.detect(img)

    if boxes is None or probs is None:
        return [], torch.empty((0, 512), device=device)

    filtered = [(b, p) for b, p in zip(boxes, probs) if p >= RUN_DETECT_PROB]
    if not filtered:
        return [], torch.empty((0, 512), device=device)

    boxes_np = np.array([f[0] for f in filtered], dtype=np.float32)
    faces    = mtcnn.extract(img, boxes_np, save_path=None)

    clean_boxes = []
    for b in boxes_np:
        left, top, right, bottom = b
        clean_boxes.append((int(top), int(right), int(bottom), int(left)))

    if faces is None or len(faces) == 0:
        return clean_boxes, torch.empty((0, 512), device=device)

    with torch.inference_mode():
        embeddings = resnet(faces.to(device))

    return clean_boxes, embeddings


# ============================================================
# FACE IDENTIFICATION
# ============================================================
def identify_faces(bgr_frame, known_emb, mtcnn, resnet, device):
    boxes, embeddings = get_faces_and_embeddings(bgr_frame, mtcnn, resnet, device)

    if len(boxes) == 0:
        return [], []
    if embeddings.numel() == 0:
        return [], boxes

    known = torch.tensor(known_emb, device=device).unsqueeze(0)
    with torch.inference_mode():
        dists = torch.norm(embeddings - known, dim=1)

    protected, strangers = [], []

    if len(dists) == 1:
        (protected if dists[0].item() <= MATCH_TOLERANCE else strangers).append(boxes[0])
        return protected, strangers

    # Multi-face: only accept best match if clearly better than second best
    order      = torch.argsort(dists)
    best_idx   = order[0].item()
    best_dist  = dists[best_idx].item()
    second_dist = dists[order[1]].item()

    for idx, box in enumerate(boxes):
        if (idx == best_idx
                and best_dist <= MATCH_TOLERANCE
                and (second_dist - best_dist) >= MATCH_MARGIN):
            protected.append(box)
        else:
            strangers.append(box)

    return protected, strangers


# ============================================================
# BLUR HELPER — ROI only (never full frame)
# ============================================================
def blur_body(output, blurred, box, shape):
    h, w = shape[:2]
    top, right, bottom, left = box
    fh = bottom - top
    fw = right  - left
    x1 = max(0, left   - int(fw * BLUR_PADDING))
    y1 = max(0, top    - int(fh * BLUR_PADDING))
    x2 = min(w, right  + int(fw * BLUR_PADDING))
    y2 = min(h, bottom + int(fh * 5.5))
    output[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
    return output


# ============================================================
# ENROLLMENT
# ============================================================
def enroll_face(cap, mtcnn, resnet, device):
    print("\n" + "=" * 55)
    print("  FACE ENROLLMENT — look straight, then tilt slightly")
    print("  left/right/up/down for better angle coverage.")
    print("=" * 55)

    for _ in range(30):
        cap.read()

    embeddings = []; attempts = 0; MAX_ATTEMPTS = 300

    while len(embeddings) < ENROLL_SAMPLES and attempts < MAX_ATTEMPTS:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            attempts += 1; continue

        bgr = cv2.flip(bgr, 1); attempts += 1
        status = "No face / look at camera"; colour = (0, 100, 255)

        try:
            img = bgr_to_pil(bgr)
            boxes, probs = mtcnn.detect(img)
            if (boxes is not None and probs is not None
                    and len(boxes) == 1 and probs[0] >= MIN_DETECT_PROB):
                faces = mtcnn.extract(img, boxes, save_path=None)
                if faces is not None and len(faces) == 1:
                    with torch.inference_mode():
                        emb = resnet(faces[0].unsqueeze(0).to(device))[0]
                    embeddings.append(emb.detach().cpu().numpy())
                    status = "Captured!"; colour = (0, 220, 100)
        except Exception as e:
            print(f"[WARN] {e}")

        display = bgr.copy()
        bar_w   = int(display.shape[1] * len(embeddings) / ENROLL_SAMPLES)
        cv2.rectangle(display, (0, 0), (display.shape[1], 55), (0,0,0), -1)
        cv2.rectangle(display, (10, 10), (10 + bar_w, 45), (0,220,100), -1)
        cv2.putText(display, f"Enrolling {len(embeddings)}/{ENROLL_SAMPLES}  |  {status}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.imshow("PrivateEye - Enrollment", display); cv2.waitKey(1)

    cv2.destroyWindow("PrivateEye - Enrollment")

    if len(embeddings) < 3:
        print("[ERROR] Not enough samples."); return None

    mean_emb = np.mean(embeddings, axis=0)
    with open(ENROLLMENT_FILE, "wb") as fh:
        pickle.dump(mean_emb, fh)
    print(f"[OK] Enrolled {len(embeddings)} samples.")
    return mean_emb


def load_enrolled_face():
    if os.path.exists(ENROLLMENT_FILE):
        with open(ENROLLMENT_FILE, "rb") as fh:
            emb = pickle.load(fh)
        print("[OK] Loaded enrolled face.")
        return emb
    return None


# ============================================================
# STARTUP MENU
# ============================================================
def startup_menu(cap, mtcnn, resnet, device):
    existing = load_enrolled_face()
    print("\n" + "="*55 + "\n  PrivateEye AI  —  Startup\n" + "="*55)
    if existing is not None:
        print("  [L]  Load enrolled face  <- recommended")
    print("  [E]  Enroll now\n  [S]  Skip (blur all faces)\n" + "="*55)
    choice = input("  Your choice: ").strip().lower()
    if choice == "e":   return enroll_face(cap, mtcnn, resnet, device)
    elif choice == "l" and existing is not None: return existing
    else: print("[INFO] Blurring all faces."); return None


# ============================================================
# BACKGROUND RECOGNITION WORKER
# Runs identify_faces() off the main thread.
# Protected-streak logic prevents flickering false unmatches.
# ============================================================
class RecognitionWorker:
    def __init__(self, known_emb, mtcnn, resnet, device):
        self.known_emb = known_emb
        self.mtcnn     = mtcnn
        self.resnet    = resnet
        self.device    = device

        self.input_frame      = None
        self.protected        = []
        self.strangers        = []
        self.protected_streak = 0
        self.protected_hold   = 0

        self.lock    = threading.Lock()
        self.event   = threading.Event()
        self.running = True
        self.thread  = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            self.event.wait(); self.event.clear()
            with self.lock: frame = self.input_frame
            if frame is None: continue
            try:
                prot, stran = identify_faces(
                    frame, self.known_emb, self.mtcnn, self.resnet, self.device)
                with self.lock:
                    if prot:
                        self.protected_streak = min(self.protected_streak + 1, STABLE_MATCH_FRAMES)
                    else:
                        self.protected_streak = max(self.protected_streak - 1, 0)

                    if self.protected_streak >= STABLE_MATCH_FRAMES:
                        self.protected      = prot
                        self.protected_hold = PROTECTED_HOLD_FRAMES
                    else:
                        if self.protected_hold > 0:
                            self.protected_hold -= 1
                        else:
                            self.protected = []

                    self.strangers = stran
            except Exception as e:
                print(f"[WARN] Recognition error: {e}")

    def submit(self, frame):
        with self.lock: self.input_frame = frame.copy()
        self.event.set()

    def get_results(self):
        with self.lock: return list(self.protected), list(self.strangers)

    def stop(self):
        self.running = False; self.event.set()


# ============================================================
# MAIN
# Pipeline:
#   ThreadedCamera     → non-blocking frame capture
#   RecognitionWorker  → GPU face ID runs off main thread
#   KalmanTracker      → predicts box positions every frame
#                        so blur always leads the face, not trails it
# ============================================================
def main():
    ensure_gpu()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    mtcnn, resnet = load_models(device)

    cap = ThreadedCamera(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera."); return

    known_emb = startup_menu(cap, mtcnn, resnet, device)
    mode      = "recognition" if known_emb is not None else "all-strangers"

    print(f"\n[INFO] Mode: {'Face Recognition' if mode == 'recognition' else 'All Strangers'}")
    print(f"[INFO] Recognize every {PROCESS_EVERY_N} frames | Kalman tracking ON")
    print("[INFO] Q: quit  R: re-enroll  C: clear\n")

    fps_timer = time.time(); frame_count = 0; fps_display = 0; frame_idx = 0

    worker  = RecognitionWorker(known_emb, mtcnn, resnet, device) if mode == "recognition" else None
    kalman  = KalmanTracker()

    # Last recognition results fed into Kalman
    last_protected = []
    last_strangers = []

    while cap.isOpened():
        ok, bgr = cap.read()
        if not ok or bgr is None: continue

        bgr = cv2.flip(bgr, 1)
        h, w = bgr.shape[:2]
        output = bgr.copy()
        blurred = None

        frame_count += 1; frame_idx += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = frame_count; frame_count = 0; fps_timer = time.time()

        # ---- FACE PRIVACY ----
        if mode == "recognition" and worker is not None:
            if frame_idx % PROCESS_EVERY_N == 0:
                worker.submit(bgr)

            new_prot, new_stran = worker.get_results()

            # Only feed Kalman new data when recognition returned something
            if new_prot != last_protected or new_stran != last_strangers:
                last_protected = new_prot
                last_strangers = new_stran
                tracked_prot, tracked_stran = kalman.update(new_prot, new_stran)
            else:
                # Recognition hasn't updated — Kalman predicts forward (no lag!)
                tracked_prot, tracked_stran = kalman.update([], [])

            # Draw green dot for protected face
            for (top, right, bottom, left) in tracked_prot:
                cv2.circle(output, ((left+right)//2, max(top-10, 10)), 8, (0,220,80), -1)

            # Blur strangers
            if tracked_stran:
                blurred = cv2.GaussianBlur(bgr, BLUR_KERNEL, 0)
                for box in tracked_stran:
                    output = blur_body(output, blurred, box, bgr.shape)

        else:
            # All-strangers mode — blur every detected face
            boxes, _ = get_faces_and_embeddings(bgr, mtcnn, resnet, device)
            if boxes:
                blurred = cv2.GaussianBlur(bgr, BLUR_KERNEL, 0)
                for box in boxes:
                    output = blur_body(output, blurred, box, bgr.shape)

        # ---- HUD ----
        mode_label = "Face Recognition" if mode == "recognition" else "All Strangers"
        cv2.rectangle(output, (0, 0), (w, 32), (0,0,0), -1)
        cv2.putText(output, f"PrivateEye  |  {mode_label}  |  {fps_display} FPS",
                    (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,80), 1)
        cv2.rectangle(output, (0, h-28), (w, h), (0,0,0), -1)
        cv2.putText(output, "Q: Quit   R: Re-enroll   C: Clear enrollment",
                    (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        cv2.imshow("PrivateEye — Active Privacy Shield", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if worker: worker.stop()
            known_emb = enroll_face(cap, mtcnn, resnet, device)
            if known_emb is not None:
                mode   = "recognition"
                worker = RecognitionWorker(known_emb, mtcnn, resnet, device)
                kalman = KalmanTracker()
                last_protected = []; last_strangers = []
                print("[INFO] Re-enrollment successful.")
        elif key == ord("c"):
            if os.path.exists(ENROLLMENT_FILE): os.remove(ENROLLMENT_FILE)
            known_emb = None
            if worker: worker.stop(); worker = None
            mode   = "all-strangers"
            kalman = KalmanTracker()
            last_protected = []; last_strangers = []
            print("[INFO] Cleared — blurring all faces.")

    if worker: worker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()