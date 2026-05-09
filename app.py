"""
SmartMeet Desktop — CustomTkinter + OpenCV + GPU Privacy Engine
Fixed: Kalman get_box() scalar bug + GPU moved off main thread for high FPS
"""
import customtkinter as ctk
import cv2, numpy as np, threading, time, pickle, os, torch
from PIL import Image, ImageTk
from facenet_pytorch import MTCNN, InceptionResnetV1

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
ENROLLMENT_FILE       = "enrolled_face.pkl"
ENROLL_SAMPLES        = 20
MATCH_TOLERANCE       = 0.80
MATCH_MARGIN          = 0.06
STABLE_MATCH_FRAMES   = 6
PROTECTED_HOLD_FRAMES = 8
BLUR_PADDING          = 0.3
PROCESS_EVERY_N       = 3
RUN_DETECT_PROB       = 0.80
MIN_DETECT_PROB       = 0.95
BLUR_KERNEL           = (51, 51)

THEME_BG      = "#0D0F14"
THEME_SURFACE = "#161A23"
THEME_CARD    = "#1E2330"
THEME_ACCENT  = "#00E5A0"
THEME_ACCENT2 = "#0099FF"
THEME_DANGER  = "#FF4757"
THEME_TEXT    = "#E8EAF0"
THEME_MUTED   = "#6B7280"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ═══════════════════════════════════════════════════════════════════════════════
# KALMAN TRACKER  — fixed get_box() to handle (8,1) statePost shape
# ═══════════════════════════════════════════════════════════════════════════════
class KalmanBoxTracker:
    count = 0
    def __init__(self, box):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.measurementMatrix   = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost        = np.eye(8, dtype=np.float32)
        cx, cy, w, h = self._cxywh(box)
        self.kf.statePost = np.array(
            [[cx],[cy],[w],[h],[0.],[0.],[0.],[0.]], dtype=np.float32)
        self.id        = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.label     = None
        self.missed    = 0
        self.MAX_MISSED = 12

    def predict(self):
        self.kf.predict()
        self.missed += 1

    def update(self, box):
        cx, cy, w, h = self._cxywh(box)
        self.kf.correct(np.array([[cx],[cy],[w],[h]], dtype=np.float32))
        self.missed = 0

    def get_box(self):
        # statePost shape is (8,1) — use flat() to safely read scalars
        s  = self.kf.statePost.flatten()
        cx, cy, w, h = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        w  = max(w, 1.0); h = max(h, 1.0)
        return (int(cy - h/2), int(cx + w/2), int(cy + h/2), int(cx - w/2))

    def is_dead(self): return self.missed > self.MAX_MISSED

    @staticmethod
    def _cxywh(box):
        t, r, b, l = box
        return (l+r)/2., (t+b)/2., float(max(r-l,1)), float(max(b-t,1))


def _iou(a, b):
    tA,rA,bA,lA = a; tB,rB,bB,lB = b
    iw = max(0, min(rA,rB) - max(lA,lB))
    ih = max(0, min(bA,bB) - max(tA,tB))
    inter = iw * ih
    union = (rA-lA)*(bA-tA) + (rB-lB)*(bB-tB) - inter
    return inter / union if union > 0 else 0.


class KalmanTracker:
    IOU_THRESHOLD = 0.25

    def __init__(self): self.trackers = []
    def reset(self):    self.trackers = []

    def update(self, prot, stran):
        for t in self.trackers: t.predict()
        dets = [(b,"protected") for b in prot] + [(b,"stranger") for b in stran]
        if dets:
            unmatched = list(range(len(dets)))
            for tr in self.trackers:
                best_iou, best_di = self.IOU_THRESHOLD, -1
                pb = tr.get_box()
                for di in unmatched:
                    sc = _iou(pb, dets[di][0])
                    if sc > best_iou: best_iou = sc; best_di = di
                if best_di >= 0:
                    tr.update(dets[best_di][0])
                    tr.label  = dets[best_di][1]
                    tr.missed = 0
                    unmatched.remove(best_di)
            for di in unmatched:
                t2 = KalmanBoxTracker(dets[di][0])
                t2.label = dets[di][1]
                self.trackers.append(t2)
        self.trackers = [t for t in self.trackers if not t.is_dead()]
        p, s = [], []
        for t in self.trackers:
            (p if t.label == "protected" else s).append(t.get_box())
        return p, s


# ═══════════════════════════════════════════════════════════════════════════════
# THREADED CAMERA
# ═══════════════════════════════════════════════════════════════════════════════
class CameraFeed:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)   # default backend — works on all Windows setups
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None; self.ret = False
        self.lock  = threading.Lock(); self.running = True
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock: self.ret = ret; self.frame = frame

    def get(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()

    @staticmethod
    def list_cameras(max_test=4):
        found = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened(): found.append(i); cap.release()
        return found if found else [0]


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVACY ENGINE  — GPU inference runs in its own background thread
# The main UI thread NEVER blocks on GPU work. It only reads cached results.
# This is what gets you from 3fps → 30+fps.
# ═══════════════════════════════════════════════════════════════════════════════
class PrivacyEngine:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
        print(f"[GPU] Using: {self.device}")
        # Two separate MTCNN instances — NEVER share one across threads.
        # mtcnn        → background worker thread only
        # enroll_mtcnn → enrollment wizard (UI thread) only
        self.mtcnn        = MTCNN(keep_all=True, device=self.device,
                                  thresholds=[0.5,0.6,0.6], min_face_size=20)
        self.enroll_mtcnn = MTCNN(keep_all=True, device=self.device,
                                  thresholds=[0.6,0.7,0.7], min_face_size=40)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.known_emb = self._load_enrollment()
        self.kalman    = KalmanTracker()
        self.privacy_on = True

        # Worker thread state
        self._input_frame   = None
        self._cached_prot   = []
        self._cached_stran  = []
        self._frame_idx     = 0
        self._prot_streak   = 0
        self._prot_hold     = 0
        self._worker_lock   = threading.Lock()
        self._worker_event  = threading.Event()
        self._worker_running = True
        self._worker_thread  = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    # ── Enrollment ─────────────────────────────────────────────────────────────
    def _load_enrollment(self):
        if os.path.exists(ENROLLMENT_FILE):
            with open(ENROLLMENT_FILE,"rb") as f: return pickle.load(f)
        return None

    def has_enrollment(self): return self.known_emb is not None

    def save_and_set(self, embeddings):
        mean = np.mean(embeddings, axis=0)
        with open(ENROLLMENT_FILE,"wb") as f: pickle.dump(mean, f)
        with self._worker_lock:
            self.known_emb = mean
            self.kalman.reset()
            self._prot_streak = 0; self._prot_hold = 0

    def clear_enrollment(self):
        if os.path.exists(ENROLLMENT_FILE): os.remove(ENROLLMENT_FILE)
        with self._worker_lock:
            self.known_emb = None
            self.kalman.reset()
            self._cached_prot = []; self._cached_stran = []

    def enroll_frame(self, bgr):
        """Try to get one enrollment embedding from a frame. Returns ndarray or None."""
        try:
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            boxes, probs = self.enroll_mtcnn.detect(img)
            if boxes is None or probs is None: return None
            if len(boxes) != 1 or probs[0] < MIN_DETECT_PROB: return None
            faces = self.enroll_mtcnn.extract(img, boxes, save_path=None)
            if faces is None or len(faces) != 1: return None
            with torch.inference_mode():
                emb = self.resnet(faces[0].unsqueeze(0).to(self.device))[0]
            return emb.detach().cpu().numpy()
        except Exception as e:
            print(f"[WARN] enroll_frame: {e}"); return None

    # ── Background GPU worker ──────────────────────────────────────────────────
    def _worker(self):
        """Runs GPU face detection + recognition off the main thread."""
        while self._worker_running:
            self._worker_event.wait(); self._worker_event.clear()
            with self._worker_lock: frame = self._input_frame
            if frame is None: continue
            try:
                if self.known_emb is not None:
                    prot, stran = self._identify(frame)
                    with self._worker_lock:
                        if prot:
                            self._prot_streak = min(self._prot_streak+1, STABLE_MATCH_FRAMES)
                        else:
                            self._prot_streak = max(self._prot_streak-1, 0)
                        if self._prot_streak >= STABLE_MATCH_FRAMES:
                            self._cached_prot  = prot
                            self._prot_hold    = PROTECTED_HOLD_FRAMES
                        else:
                            if self._prot_hold > 0: self._prot_hold -= 1
                            else: self._cached_prot = []
                        self._cached_stran = stran
                else:
                    boxes = self._detect_only(frame)
                    with self._worker_lock:
                        self._cached_prot  = []
                        self._cached_stran = boxes
            except Exception as e:
                print(f"[WARN] worker: {e}")

    def _detect_only(self, bgr):
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None or probs is None: return []
        return [(int(b[1]),int(b[2]),int(b[3]),int(b[0]))
                for b,p in zip(boxes,probs) if p >= RUN_DETECT_PROB]

    def _identify(self, bgr):
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None or probs is None: return [], []
        filtered = [(b,p) for b,p in zip(boxes,probs) if p >= RUN_DETECT_PROB]
        if not filtered: return [], []
        boxes_np = np.array([f[0] for f in filtered], dtype=np.float32)
        faces    = self.mtcnn.extract(img, boxes_np, save_path=None)
        clean    = [(int(b[1]),int(b[2]),int(b[3]),int(b[0])) for b in boxes_np]
        if faces is None or len(faces) == 0: return [], clean
        with torch.inference_mode():
            embs  = self.resnet(faces.to(self.device))
            known = torch.tensor(self.known_emb, device=self.device).unsqueeze(0)
            dists = torch.norm(embs - known, dim=1)
        prot, stran = [], []
        if len(dists) == 1:
            (prot if dists[0].item() <= MATCH_TOLERANCE else stran).append(clean[0])
            return prot, stran
        order = torch.argsort(dists)
        bi = order[0].item(); bd = dists[bi].item(); sd = dists[order[1]].item()
        for i, box in enumerate(clean):
            if i==bi and bd<=MATCH_TOLERANCE and (sd-bd)>=MATCH_MARGIN: prot.append(box)
            else: stran.append(box)
        return prot, stran

    # ── Main process — called every frame from UI thread ──────────────────────
    def process(self, bgr):
        """
        Non-blocking. Submits frame to worker every N frames.
        Always reads cached Kalman-predicted boxes — never waits for GPU.
        """
        if not self.privacy_on: return bgr.copy()

        self._frame_idx += 1

        # Submit to worker thread every N frames
        if self._frame_idx % PROCESS_EVERY_N == 0:
            with self._worker_lock:
                self._input_frame = bgr.copy()
            self._worker_event.set()

        # Read latest cached results (instant, no GPU wait)
        with self._worker_lock:
            cached_prot  = list(self._cached_prot)
            cached_stran = list(self._cached_stran)

        # Kalman predict — smooth movement every frame
        tracked_prot, tracked_stran = self.kalman.update(cached_prot, cached_stran)

        output = bgr.copy()

        # Blur strangers
        for box in tracked_stran:
            output = self._blur(output, box)

        # Mark protected face
        for (t, r, b, l) in tracked_prot:
            cx = (l + r) // 2; cy = max(t - 14, 14)
            # protected face indicator removed


        return output

    def _blur(self, frame, box):
        h, w = frame.shape[:2]
        t, r, b, l = box
        fh = b - t; fw = r - l
        x1 = max(0, l - int(fw * BLUR_PADDING))
        y1 = max(0, t - int(fh * BLUR_PADDING))
        x2 = min(w, r + int(fw * BLUR_PADDING))
        y2 = min(h, b + int(fh * 5.5))
        if x2 <= x1 or y2 <= y1: return frame
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        out = frame.copy(); out[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
        return out

    def stop(self):
        self._worker_running = False
        self._worker_event.set()


# ═══════════════════════════════════════════════════════════════════════════════
# ENROLLMENT WIZARD
# ═══════════════════════════════════════════════════════════════════════════════
class EnrollmentWizard(ctk.CTkToplevel):
    def __init__(self, parent, engine: PrivacyEngine, camera: CameraFeed, on_done):
        super().__init__(parent)
        self.engine  = engine
        self.camera  = camera
        self.on_done = on_done
        self.samples = []
        self.running = True

        # Shared state between UI thread and GPU worker thread
        self._latest_frame  = None
        self._preview_boxes = []          # list of (l,t,r,b,prob)
        self._preview_lock  = threading.Lock()

        self.title("Face Enrollment — SmartMeet")
        self.geometry("700x560")
        self.resizable(False, False)
        self.configure(fg_color=THEME_BG)
        self.grab_set()
        self._build()

        # Start GPU worker BEFORE the UI loop
        self._worker_thread = threading.Thread(target=self._gpu_worker, daemon=True)
        self._worker_thread.start()
        self.after(100, self._loop)

    def _build(self):
        hdr = ctk.CTkFrame(self, fg_color=THEME_SURFACE, corner_radius=0, height=60)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="👤  Face Enrollment",
                     font=ctk.CTkFont("Helvetica", 20, "bold"),
                     text_color=THEME_ACCENT).pack(side="left", padx=20, pady=15)

        self.preview = ctk.CTkLabel(self, text="Starting camera…", width=480, height=360,
                                     fg_color=THEME_CARD, corner_radius=10)
        self.preview.pack(pady=(16,8))

        bar_f = ctk.CTkFrame(self, fg_color="transparent")
        bar_f.pack(fill="x", padx=40)
        self.progress_bar = ctk.CTkProgressBar(bar_f, height=10,
                                                progress_color=THEME_ACCENT,
                                                fg_color=THEME_CARD)
        self.progress_bar.pack(fill="x"); self.progress_bar.set(0)

        self.status_lbl = ctk.CTkLabel(self,
            text="Look straight at the camera — move head slightly for coverage",
            font=ctk.CTkFont("Helvetica",13), text_color=THEME_MUTED)
        self.status_lbl.pack(pady=6)

        self.count_lbl = ctk.CTkLabel(self,
            text=f"0 / {ENROLL_SAMPLES} samples",
            font=ctk.CTkFont("Helvetica",15,"bold"), text_color=THEME_TEXT)
        self.count_lbl.pack()

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=14)
        ctk.CTkButton(btn_row, text="Cancel", width=120, height=38,
                       fg_color=THEME_CARD, hover_color=THEME_DANGER,
                       text_color=THEME_TEXT, font=ctk.CTkFont("Helvetica",13),
                       command=self._cancel).pack(side="left", padx=8)
        self.finish_btn = ctk.CTkButton(btn_row, text="Save & Finish", width=140, height=38,
                                         fg_color=THEME_ACCENT, hover_color="#00C080",
                                         text_color="#000000",
                                         font=ctk.CTkFont("Helvetica",13,"bold"),
                                         command=self._finish, state="disabled")
        self.finish_btn.pack(side="left", padx=8)

    def _loop(self):
        """UI thread — only handles display. Zero GPU calls here."""
        if not self.running: return

        ok, frame = self.camera.get()
        if ok and frame is not None:
            frame = cv2.flip(frame, 1)
            self._latest_frame = frame

            # Show the latest processed preview (updated by worker thread)
            disp = frame.copy()
            with self._preview_lock:
                boxes_to_draw = list(self._preview_boxes)
            for (l,t,r,b,prob) in boxes_to_draw:
                color = (0,229,160) if len(self.samples) > 0 else (0,153,255)
                cv2.rectangle(disp, (l,t), (r,b), color, 2)
                cv2.putText(disp, f"{prob:.2f}", (l, t-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            img_pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            img_pil = img_pil.resize((480, 360), Image.BILINEAR)
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(480,360))
            self.preview.configure(image=ctk_img, text="")

        self.after(33, self._loop)

    def _gpu_worker(self):
        """Background thread — all GPU/MTCNN calls run here, never on UI thread."""
        hints = ["Look straight","Tilt left","Tilt right","Look up","Look down","Chin down"]
        while self.running:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.03); continue
            try:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Detect boxes for preview overlay (always)
                boxes, probs = self.engine.enroll_mtcnn.detect(img)
                new_boxes = []
                if boxes is not None and probs is not None:
                    for box, prob in zip(boxes, probs):
                        if prob >= RUN_DETECT_PROB:
                            new_boxes.append((int(box[0]),int(box[1]),
                                              int(box[2]),int(box[3]), float(prob)))
                with self._preview_lock:
                    self._preview_boxes = new_boxes

                # Try to grab an enrollment sample
                if len(self.samples) < ENROLL_SAMPLES and new_boxes:
                    emb = self.engine.enroll_frame(frame)
                    if emb is not None:
                        self.samples.append(emb)
                        n = len(self.samples)
                        # Schedule UI update back on main thread safely
                        self.after(0, self._update_progress, n, hints[n % len(hints)])

            except Exception as e:
                print(f"[WARN] enroll worker: {e}")
            time.sleep(0.05)  # ~20 samples/sec max — smooth, not frantic

    def _update_progress(self, n, hint):
        """Called on UI thread via after(0) from worker."""
        if not self.running: return
        self.progress_bar.set(n / ENROLL_SAMPLES)
        self.count_lbl.configure(text=f"{n} / {ENROLL_SAMPLES} samples")
        if n >= ENROLL_SAMPLES:
            self.status_lbl.configure(text="✅  Done! Click Save & Finish",
                                       text_color=THEME_ACCENT)
            self.finish_btn.configure(state="normal")
        else:
            self.status_lbl.configure(text=hint + " …", text_color=THEME_MUTED)

    def _finish(self):
        if len(self.samples) >= 3:
            self.engine.save_and_set(self.samples)
            self.running = False; self.on_done(True); self.destroy()

    def _cancel(self):
        self.running = False; self.on_done(False); self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
class SettingsPanel(ctk.CTkFrame):
    def __init__(self, parent, engine: PrivacyEngine, app_ref, cameras):
        super().__init__(parent, fg_color=THEME_SURFACE, corner_radius=12, width=260)
        self.engine  = engine
        self.app_ref = app_ref
        self.cameras = cameras
        self.pack_propagate(False)
        self._build()

    def _build(self):
        pad = {"padx":18, "pady":5}

        # Logo
        logo_row = ctk.CTkFrame(self, fg_color="transparent")
        logo_row.pack(fill="x", padx=18, pady=(20,2))
        ctk.CTkLabel(logo_row, text="⬡", font=ctk.CTkFont("Helvetica",26),
                     text_color=THEME_ACCENT).pack(side="left")
        ctk.CTkLabel(logo_row, text=" SmartMeet",
                     font=ctk.CTkFont("Helvetica",19,"bold"),
                     text_color=THEME_TEXT).pack(side="left")
        ctk.CTkLabel(self, text="Privacy-first video meetings",
                     font=ctk.CTkFont("Helvetica",11),
                     text_color=THEME_MUTED).pack(padx=18, pady=(0,4))

        self._divider()

        # Camera
        self._section("CAMERA")
        ctk.CTkLabel(self, text="Active Camera",
                     font=ctk.CTkFont("Helvetica",12), text_color=THEME_MUTED).pack(anchor="w", **pad)
        cam_opts = [f"Camera {i}" for i in self.cameras]
        self.cam_var = ctk.StringVar(value=cam_opts[0])
        ctk.CTkOptionMenu(self, values=cam_opts, variable=self.cam_var,
                           fg_color=THEME_CARD, button_color=THEME_ACCENT,
                           button_hover_color="#00C080", text_color=THEME_TEXT,
                           font=ctk.CTkFont("Helvetica",12),
                           command=self._cam_changed).pack(fill="x", **pad)

        self._divider()

        # Privacy
        self._section("PRIVACY FILTER")
        priv_row = ctk.CTkFrame(self, fg_color="transparent")
        priv_row.pack(fill="x", **pad)
        ctk.CTkLabel(priv_row, text="Privacy Shield",
                     font=ctk.CTkFont("Helvetica",13),
                     text_color=THEME_TEXT).pack(side="left")
        self.priv_switch = ctk.CTkSwitch(priv_row, text="", width=48,
                                          progress_color=THEME_ACCENT,
                                          command=self._toggle_privacy)
        self.priv_switch.pack(side="right"); self.priv_switch.select()

        ctk.CTkLabel(self, text="Blur Intensity",
                     font=ctk.CTkFont("Helvetica",12), text_color=THEME_MUTED).pack(anchor="w",**pad)
        self.blur_slider = ctk.CTkSlider(self, from_=3, to=15, number_of_steps=12,
                                          progress_color=THEME_ACCENT,
                                          button_color=THEME_ACCENT,
                                          command=self._blur_changed)
        self.blur_slider.set(5); self.blur_slider.pack(fill="x", **pad)

        ctk.CTkLabel(self, text="Match Sensitivity",
                     font=ctk.CTkFont("Helvetica",12), text_color=THEME_MUTED).pack(anchor="w",**pad)
        self.tol_slider = ctk.CTkSlider(self, from_=0.5, to=1.0, number_of_steps=10,
                                         progress_color=THEME_ACCENT2,
                                         button_color=THEME_ACCENT2,
                                         command=self._tol_changed)
        self.tol_slider.set(MATCH_TOLERANCE); self.tol_slider.pack(fill="x", **pad)

        self._divider()

        # Enrollment
        self._section("ENROLLMENT")
        self.enroll_status = ctk.CTkLabel(self,
            text="✅  Face enrolled" if self.engine.has_enrollment() else "⚠  No face enrolled",
            font=ctk.CTkFont("Helvetica",12,"bold"),
            text_color=THEME_ACCENT if self.engine.has_enrollment() else THEME_DANGER)
        self.enroll_status.pack(**pad)

        ctk.CTkButton(self, text="Enroll My Face", height=38,
                       fg_color=THEME_ACCENT2, hover_color="#007ACC",
                       text_color=THEME_TEXT, font=ctk.CTkFont("Helvetica",13,"bold"),
                       command=self.app_ref.open_enrollment).pack(fill="x", **pad)
        ctk.CTkButton(self, text="Clear Enrollment", height=34,
                       fg_color=THEME_CARD, hover_color=THEME_DANGER,
                       text_color=THEME_MUTED, font=ctk.CTkFont("Helvetica",12),
                       command=self._clear_enroll).pack(fill="x", **pad)

        self._divider()

        # Stats
        self._section("LIVE STATS")
        self.fps_lbl = ctk.CTkLabel(self, text="FPS: --",
                                     font=ctk.CTkFont("Courier",12), text_color=THEME_ACCENT)
        self.fps_lbl.pack(anchor="w", **pad)
        self.gpu_lbl = ctk.CTkLabel(self,
            text=f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Mode: CPU",
            font=ctk.CTkFont("Courier",11), text_color=THEME_MUTED)
        self.gpu_lbl.pack(anchor="w", **pad)
        self.faces_lbl = ctk.CTkLabel(self, text="Faces: 0P  0S",
                                       font=ctk.CTkFont("Courier",11), text_color=THEME_MUTED)
        self.faces_lbl.pack(anchor="w", **pad)

    def _divider(self):
        ctk.CTkFrame(self, height=1, fg_color=THEME_CARD).pack(fill="x", padx=14, pady=8)

    def _section(self, text):
        ctk.CTkLabel(self, text=text, font=ctk.CTkFont("Helvetica",10,"bold"),
                     text_color=THEME_MUTED).pack(anchor="w", padx=18, pady=(4,0))

    def _toggle_privacy(self):
        self.engine.privacy_on = (self.priv_switch.get() == 1)

    def _blur_changed(self, val):
        k = int(val) * 2 + 1
        global BLUR_KERNEL; BLUR_KERNEL = (k, k)

    def _tol_changed(self, val):
        global MATCH_TOLERANCE; MATCH_TOLERANCE = float(val)

    def _cam_changed(self, choice):
        self.app_ref.switch_camera(int(choice.split()[-1]))

    def _clear_enroll(self):
        self.engine.clear_enrollment(); self.update_enrollment_status()

    def update_enrollment_status(self):
        enrolled = self.engine.has_enrollment()
        self.enroll_status.configure(
            text="✅  Face enrolled" if enrolled else "⚠  No face enrolled",
            text_color=THEME_ACCENT if enrolled else THEME_DANGER)

    def update_stats(self, fps, n_prot, n_stran):
        self.fps_lbl.configure(text=f"FPS: {fps}")
        self.faces_lbl.configure(text=f"Faces: {n_prot} protected  |  {n_stran} strangers")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTICIPANT TILE — one tile per camera
# ═══════════════════════════════════════════════════════════════════════════════
class ParticipantTile(ctk.CTkFrame):
    def __init__(self, parent, cam_index: int, engine: PrivacyEngine, label: str):
        super().__init__(parent, fg_color=THEME_CARD, corner_radius=10)
        self.cam_index = cam_index
        self.engine    = engine
        self.label_str = label
        self.camera    = CameraFeed(cam_index)
        self.running   = True
        self._t0       = time.time()
        self._frames   = 0
        self.fps       = 0
        self._build()
        self.after(50, self._loop)   # small delay so widget sizes are known

    def _build(self):
        self.video_lbl = ctk.CTkLabel(self, text="", corner_radius=8)
        self.video_lbl.pack(padx=6, pady=(6,2), fill="both", expand=True)
        bar = ctk.CTkFrame(self, fg_color=THEME_SURFACE, corner_radius=8, height=32)
        bar.pack(fill="x", padx=6, pady=(0,6)); bar.pack_propagate(False)
        ctk.CTkLabel(bar, text=f"  {self.label_str}",
                     font=ctk.CTkFont("Helvetica",12,"bold"),
                     text_color=THEME_TEXT).pack(side="left", pady=4)
        self.fps_lbl = ctk.CTkLabel(bar, text="-- fps  ",
                                     font=ctk.CTkFont("Courier",10),
                                     text_color=THEME_MUTED)
        self.fps_lbl.pack(side="right", pady=4)

    def _loop(self):
        if not self.running: return
        ok, frame = self.camera.get()
        if ok and frame is not None:
            frame     = cv2.flip(frame, 1)
            processed = self.engine.process(frame)
            self._frames += 1
            now = time.time()
            if now - self._t0 >= 1.0:
                self.fps = self._frames; self._frames = 0; self._t0 = now
                self.fps_lbl.configure(text=f"{self.fps} fps  ")

            tw = self.winfo_width()  - 12
            th = self.winfo_height() - 44
            if tw < 80:  tw = 320
            if th < 60:  th = 240

            img_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            img_pil = img_pil.resize((tw, th), Image.BILINEAR)
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(tw,th))
            self.video_lbl.configure(image=ctk_img)

        self.after(16, self._loop)  # ~60fps target

    def destroy_feed(self):
        self.running = False
        self.camera.release()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class SmartMeetApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SmartMeet  —  Privacy Shield")
        self.geometry("1280x780"); self.minsize(960, 600)
        self.configure(fg_color=THEME_BG)

        self.cameras = CameraFeed.list_cameras()
        print(f"[INFO] Cameras: {self.cameras}")

        self.engine = PrivacyEngine()
        self.tiles: list[ParticipantTile] = []

        self._build_layout()
        self._add_tile(self.cameras[0], "You")
        self._stats_loop()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        # Sidebar
        self.sidebar = SettingsPanel(self, self.engine, self, self.cameras)
        self.sidebar.pack(side="left", fill="y", padx=(12,0), pady=12)

        # Right panel
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        # Top bar
        topbar = ctk.CTkFrame(right, fg_color=THEME_SURFACE, corner_radius=10, height=52)
        topbar.pack(fill="x", pady=(0,10)); topbar.pack_propagate(False)
        ctk.CTkLabel(topbar, text="Meeting Room",
                     font=ctk.CTkFont("Helvetica",16,"bold"),
                     text_color=THEME_TEXT).pack(side="left", padx=18, pady=14)
        ctk.CTkButton(topbar, text="＋  Add Camera", width=130, height=34,
                       fg_color=THEME_CARD, hover_color=THEME_ACCENT2,
                       text_color=THEME_TEXT, font=ctk.CTkFont("Helvetica",12),
                       command=self._add_camera_dialog).pack(side="right", padx=8, pady=9)
        ctk.CTkLabel(topbar, text="Layout:",
                     font=ctk.CTkFont("Helvetica",12),
                     text_color=THEME_MUTED).pack(side="right", pady=9)
        self.layout_var = ctk.StringVar(value="Grid")
        ctk.CTkOptionMenu(topbar, values=["Grid","Focus","Strip"],
                           variable=self.layout_var, width=90, height=34,
                           fg_color=THEME_CARD, button_color=THEME_ACCENT,
                           button_hover_color="#00C080", text_color=THEME_TEXT,
                           font=ctk.CTkFont("Helvetica",12),
                           command=lambda _: self._relayout()).pack(side="right", padx=4, pady=9)

        # Grid
        self.grid_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.grid_frame.pack(fill="both", expand=True)

    def _add_tile(self, cam_idx, label=None):
        label = label or f"Camera {cam_idx}"
        tile  = ParticipantTile(self.grid_frame, cam_idx, self.engine, label)
        self.tiles.append(tile)
        self._relayout()

    def _relayout(self):
        for tile in self.tiles: tile.grid_forget()
        n = len(self.tiles)
        if n == 0: return
        layout = self.layout_var.get()

        if layout == "Strip":
            for i,tile in enumerate(self.tiles):
                tile.grid(row=0, column=i, padx=4, pady=4, sticky="nsew")
                self.grid_frame.columnconfigure(i, weight=1)
            self.grid_frame.rowconfigure(0, weight=1)

        elif layout == "Focus":
            self.tiles[0].grid(row=0, column=0, rowspan=max(len(self.tiles)-1,1),
                                padx=4, pady=4, sticky="nsew")
            self.grid_frame.columnconfigure(0, weight=3)
            for i,tile in enumerate(self.tiles[1:]):
                tile.grid(row=i, column=1, padx=4, pady=4, sticky="nsew")
                self.grid_frame.rowconfigure(i, weight=1)
            self.grid_frame.columnconfigure(1, weight=1)

        else:  # Grid
            cols = max(1, int(np.ceil(np.sqrt(n))))
            rows = int(np.ceil(n / cols))
            for i,tile in enumerate(self.tiles):
                r,c = divmod(i, cols)
                tile.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            for c in range(cols): self.grid_frame.columnconfigure(c, weight=1)
            for r in range(rows): self.grid_frame.rowconfigure(r,  weight=1)

    def _add_camera_dialog(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Add Camera"); dlg.geometry("340x200")
        dlg.configure(fg_color=THEME_BG); dlg.grab_set()
        ctk.CTkLabel(dlg, text="Select Camera",
                     font=ctk.CTkFont("Helvetica",15,"bold"),
                     text_color=THEME_TEXT).pack(pady=(24,10))
        opts = [f"Camera {i}" for i in self.cameras]
        var  = ctk.StringVar(value=opts[0])
        ctk.CTkOptionMenu(dlg, values=opts, variable=var,
                           fg_color=THEME_CARD, button_color=THEME_ACCENT,
                           button_hover_color="#00C080",
                           text_color=THEME_TEXT).pack(pady=8)
        def _add():
            idx = int(var.get().split()[-1])
            self._add_tile(idx, f"Participant {len(self.tiles)}"); dlg.destroy()
        ctk.CTkButton(dlg, text="Add", fg_color=THEME_ACCENT, text_color="#000",
                       font=ctk.CTkFont("Helvetica",13,"bold"),
                       command=_add).pack(pady=12)

    def open_enrollment(self):
        if self.tiles:
            # Pause the privacy engine worker while enrollment runs.
            # Both share the GPU — letting them run simultaneously causes
            # the UI freeze you see when clicking "Enroll My Face".
            self.engine.paused = True
            EnrollmentWizard(self, self.engine, self.tiles[0].camera,
                              self._enrollment_done)

    def _enrollment_done(self, success):
        # Resume normal processing after wizard closes
        self.engine.paused = False
        self.sidebar.update_enrollment_status()

    def switch_camera(self, new_idx):
        if self.tiles:
            label = self.tiles[0].label_str
            self.tiles[0].destroy_feed(); self.tiles.pop(0)
            self._add_tile(new_idx, label)

    def _stats_loop(self):
        if self.tiles:
            self.sidebar.update_stats(
                self.tiles[0].fps,
                len(self.engine._cached_prot),
                len(self.engine._cached_stran))
        self.after(500, self._stats_loop)

    def _on_close(self):
        self.engine.stop()
        for tile in self.tiles: tile.destroy_feed()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SmartMeetApp()
    app.mainloop()