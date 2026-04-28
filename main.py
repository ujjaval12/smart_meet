import cv2
import numpy as np
import os
import pickle
import time
import tempfile

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def ensure_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA GPU is required. No GPU detected.")


def load_models(device):
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet


def bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# ============================================================
# CONFIG
# ============================================================
ENROLLMENT_FILE = "enrolled_face.pkl"
ENROLL_SAMPLES = 10
MATCH_TOLERANCE = 0.75
BLUR_PADDING = 0.3
MIN_DETECT_PROB = 0.95


def get_faces_and_embeddings(bgr, mtcnn, resnet, device):
    img = bgr_to_pil(bgr)
    boxes, probs = mtcnn.detect(img)
    if boxes is None or probs is None:
        return [], torch.empty((0, 512), device=device)

    faces = mtcnn.extract(img, boxes, save_path=None)
    if faces is None or len(faces) == 0:
        return [], torch.empty((0, 512), device=device)

    faces = faces.to(device)
    with torch.inference_mode():
        embeddings = resnet(faces)

    clean_boxes = []
    for box in boxes:
        left, top, right, bottom = box
        clean_boxes.append((int(top), int(right), int(bottom), int(left)))

    return clean_boxes, embeddings


# ============================================================
# FACE ENROLLMENT
# ============================================================
def enroll_face(cap, mtcnn, resnet, device):
    print("\n" + "=" * 55)
    print("  FACE ENROLLMENT MODE")
    print("  Look straight at the camera.")
    print("  Hold still — capturing reference frames...")
    print("=" * 55)

    # Flush stale frames from buffer
    for _ in range(30):
        cap.read()

    embeddings   = []
    attempts     = 0
    MAX_ATTEMPTS = 200

    while len(embeddings) < ENROLL_SAMPLES and attempts < MAX_ATTEMPTS:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            attempts += 1
            continue

        bgr = cv2.flip(bgr, 1)
        attempts += 1

        try:
            img = bgr_to_pil(bgr)
            boxes, probs = mtcnn.detect(img)
            if boxes is not None and probs is not None and len(boxes) == 1 and probs[0] >= MIN_DETECT_PROB:
                faces = mtcnn.extract(img, boxes, save_path=None)
                if faces is not None and len(faces) == 1:
                    face = faces[0].unsqueeze(0).to(device)
                    with torch.inference_mode():
                        emb = resnet(face)[0]
                    embeddings.append(emb.detach().cpu().numpy())
        except Exception as e:
            print(f"[WARN] Detection/encoding failed: {e}")

        # Progress UI — drawn on a display copy, never on bgr
        display  = bgr.copy()
        progress = len(embeddings) / ENROLL_SAMPLES
        bar_w    = int(display.shape[1] * progress)
        cv2.rectangle(display, (0, 0), (display.shape[1], 55), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (10 + bar_w, 45), (0, 220, 100), -1)
        status = "Face detected!" if len(embeddings) > 0 else "No face / look at camera"
        colour = (0, 220, 100) if len(embeddings) > 0 else (0, 100, 255)
        cv2.putText(display,
                    f"Enrolling {len(embeddings)}/{ENROLL_SAMPLES}  |  {status}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.imshow("PrivateEye - Enrollment", display)
        cv2.waitKey(1)

    cv2.destroyWindow("PrivateEye - Enrollment")

    if len(embeddings) < 3:
        print("[ERROR] Not enough samples captured. Try better lighting.")
        return None

    mean_emb = np.mean(embeddings, axis=0)
    with open(ENROLLMENT_FILE, "wb") as fh:
        pickle.dump(mean_emb, fh)
    print(f"[OK] Enrolled {len(embeddings)} samples → saved to '{ENROLLMENT_FILE}'")
    return mean_emb


def load_enrolled_face():
    if os.path.exists(ENROLLMENT_FILE):
        with open(ENROLLMENT_FILE, "rb") as fh:
            emb = pickle.load(fh)
        print(f"[OK] Loaded enrolled face from '{ENROLLMENT_FILE}'")
        return emb
    return None


# ============================================================
# FACE IDENTIFICATION
# ============================================================
def identify_faces(bgr_frame, known_emb, mtcnn, resnet, device):
    boxes, embeddings = get_faces_and_embeddings(bgr_frame, mtcnn, resnet, device)
    if len(boxes) == 0:
        return [], []

    known = torch.tensor(known_emb, device=device).unsqueeze(0)
    protected, strangers = [], []

    with torch.inference_mode():
        dists = torch.norm(embeddings - known, dim=1)

    for box, dist in zip(boxes, dists):
        if dist.item() <= MATCH_TOLERANCE:
            protected.append(box)
        else:
            strangers.append(box)

    return protected, strangers


# ============================================================
# BLUR HELPERS
# ============================================================
def blur_body(output, blurred, box, shape):
    h, w = shape[:2]
    top, right, bottom, left = box
    fh = bottom - top
    fw = right - left
    x1 = max(0, left   - int(fw * BLUR_PADDING))
    y1 = max(0, top    - int(fh * BLUR_PADDING))
    x2 = min(w, right  + int(fw * BLUR_PADDING))
    y2 = min(h, bottom + int(fh * 5.5))
    output[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
    return output


def startup_menu(cap, mtcnn, resnet, device):
    existing = load_enrolled_face()

    print("\n" + "=" * 55)
    print("  PrivateEye AI  —  Startup")
    print("=" * 55)
    if existing is not None:
        print("  [L]  Load previously enrolled face  <- recommended")
    print("  [E]  Enroll a new face now")
    print("  [S]  Skip  (treat all faces as strangers)")
    print("=" * 55)

    choice = input("  Your choice: ").strip().lower()

    if choice == "e":
        return enroll_face(cap, mtcnn, resnet, device)
    elif choice == "l" and existing is not None:
        return existing
    else:
        print("[INFO] Running with all faces treated as strangers.")
        return None


# ============================================================
# STARTUP MENU
# ============================================================
# ============================================================
# MAIN
# ============================================================
def main():
    ensure_gpu()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    mtcnn, resnet = load_models(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    known_emb = startup_menu(cap, mtcnn, resnet, device)
    mode = "recognition" if known_emb is not None else "all-strangers"

    print(f"\n[INFO] Mode: {'Face Recognition' if mode == 'recognition' else 'All Strangers'}")
    print("[INFO] Running — Q: quit  |  R: re-enroll  |  C: clear enrollment\n")

    fps_timer    = time.time()
    frame_count  = 0
    fps_display  = 0
    skip_counter = 0          # process face recognition every N frames
    SKIP_FRAMES  = 2          # run FR every 3rd frame, draw cached result in between
    cached_protected = []
    cached_strangers = []

    while cap.isOpened():
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue

        bgr     = cv2.flip(bgr, 1)
        h, w    = bgr.shape[:2]
        output  = bgr.copy()
        blurred = cv2.GaussianBlur(bgr, (99, 99), 0)

        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = frame_count
            frame_count = 0
            fps_timer   = time.time()

        # ---- FEATURE 1: FACE PRIVACY ----
        if mode == "recognition":
            try:
                # Only run heavy FR every SKIP_FRAMES frames
                skip_counter += 1
                if skip_counter > SKIP_FRAMES:
                    skip_counter = 0
                    cached_protected, cached_strangers = identify_faces(bgr, known_emb, mtcnn, resnet, device)

                for (top, right, bottom, left) in cached_protected:
                    cx = (left + right) // 2
                    cv2.circle(output, (cx, max(top - 10, 10)), 8, (0, 220, 80), -1)
                for box in cached_strangers:
                    output = blur_body(output, blurred, box, bgr.shape)
            except Exception as e:
                print(f"[WARN] Recognition error: {e}")
        else:
            boxes, _ = get_faces_and_embeddings(bgr, mtcnn, resnet, device)
            if boxes:
                for box in boxes:
                    output = blur_body(output, blurred, box, bgr.shape)

        # ---- HUD ----
        mode_label = "Face Recognition" if mode == "recognition" else "All Strangers"
        cv2.rectangle(output, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.putText(output, f"PrivateEye  |  {mode_label}  |  {fps_display} FPS",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 1)
        cv2.rectangle(output, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.putText(output, "Q: Quit   R: Re-enroll   C: Clear enrollment",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("PrivateEye — Active Privacy Shield", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            known_emb = enroll_face(cap, mtcnn, resnet, device)
            if known_emb is not None:
                mode = "recognition"
                print("[INFO] Re-enrollment successful.")
        elif key == ord("c"):
            if os.path.exists(ENROLLMENT_FILE):
                os.remove(ENROLLMENT_FILE)
            known_emb = None
                mode = "all-strangers"
                print("[INFO] Enrollment cleared — switched to all-strangers mode.")
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()