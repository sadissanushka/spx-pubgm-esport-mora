import os
import re
import sys
import time
import logging
import cv2
import numpy as np
import mss
import pygetwindow as gw

lib_path = r'C:\Users\ASUS-TUF\lib'
if os.path.exists(lib_path):
    sys.path.insert(0, lib_path)

try:
    import torch
    import easyocr
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — check driver/PyTorch install")
except ImportError as e:
    print(f"Missing dependency: {e}\n  pip install torch easyocr")
    sys.exit(1)
except RuntimeError as e:
    print(f"GPU error: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ---------------------------------------------------------------------------
# Emulator window titles to search (in priority order)
# ---------------------------------------------------------------------------
EMULATOR_TITLES = ["Gameloop", "PUBG MOBILE", "BlueStacks", "LDPlayer"]

# ---------------------------------------------------------------------------
# HSV color ranges — tuned to exact HUD colors
#
#   STANDBY  #C3C0AF  →  H≈42°  S≈12%  V≈76%
#             OpenCV:    H≈21   S≈31   V≈195
#
#   CLOSING  #FF3055  →  H≈349° S≈81%  V≈100%
#             OpenCV:    H≈175  S≈207  V≈255
#             (wraps:    H≈0-3  also valid)
#
# Ranges are intentionally tight to avoid false matches from other HUD
# elements (health bars, minimap, kill feed).
# ---------------------------------------------------------------------------

# CLOSING — high-saturation red-pink, two bands because hue wraps at 180
RED_LOWER1 = np.array([0,   160, 200])   # 0–5°  (wrap-around band)
RED_UPPER1 = np.array([5,   255, 255])
RED_LOWER2 = np.array([168, 160, 200])   # 336–360° band (CV: 168-179)
RED_UPPER2 = np.array([179, 255, 255])

# STANDBY — low-saturation warm gray
GRAY_LOWER = np.array([15,  10,  160])   # H 30-55°, very low S, mid-high V
GRAY_UPPER = np.array([30,  50,  220])

# Red must exceed this pixel count AND dominate gray to confirm CLOSING
RED_THRESHOLD = 12    # small — icon can be as few as ~15×15 px

# Frame must differ by this mean-abs-diff to bother re-running OCR
FRAME_DIFF_THRESHOLD = 5.0


class PUBGTracker:
    def __init__(self):
        logging.info("Initializing EasyOCR on CUDA…")
        # model_storage_directory keeps weights local so re-init is instant
        self.reader = easyocr.Reader(
            ['en'],
            gpu=True,
            model_storage_directory=os.path.join(os.path.dirname(__file__), '.easyocr'),
            verbose=False,
        )
        logging.info(f"EasyOCR ready  —  GPU: {torch.cuda.get_device_name(0)}")

        self.region = None
        self.state  = "STANDBY"
        self.current_stage    = None
        self.current_time_str = ""
        self.end_time         = 0.0

        self._prev_gray   = None   # frame-diff baseline
        self._state_votes = []     # debounce ring buffer
        self._DEBOUNCE    = 3      # consecutive frames needed to flip state

    # -----------------------------------------------------------------------
    # Window detection
    # -----------------------------------------------------------------------
    def find_window(self) -> bool:
        for title in EMULATOR_TITLES:
            wins = gw.getWindowsWithTitle(title)
            if wins:
                win = wins[0]
                logging.info(f"Window '{win.title}'  {win.width}x{win.height}  @({win.left},{win.top})")
                self.region = {
                    "top":    win.top  + int(win.height * 0.03),
                    "left":   win.left + int(win.width  * 0.83),
                    "width":  int(win.width  * 0.14),
                    "height": int(win.height * 0.07),
                }
                logging.info(f"Capture region: {self.region}")
                return True
        logging.warning("Game window not found. Retrying in 5 s…")
        return False

    # -----------------------------------------------------------------------
    # Color-based state detection (CPU — instant, no GPU overhead needed)
    # -----------------------------------------------------------------------
    def _detect_state(self, bgr: np.ndarray) -> str:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        red_mask = (cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) |
                    cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
        # Erode 1px to kill single-pixel noise before counting
        red_mask = cv2.erode(red_mask, np.ones((2, 2), np.uint8), iterations=1)
        red  = cv2.countNonZero(red_mask)
        gray = cv2.countNonZero(cv2.inRange(hsv, GRAY_LOWER, GRAY_UPPER))

        logging.debug(f"Color px — red:{red}  gray:{gray}")

        # CLOSING: enough red AND red pixels dominate gray
        if red >= RED_THRESHOLD and red > gray * 0.25:
            return "CLOSING"
        return "STANDBY"

    def _debounced_state(self, raw: str) -> str:
        self._state_votes.append(raw)
        if len(self._state_votes) > self._DEBOUNCE:
            self._state_votes.pop(0)
        if self._state_votes.count(raw) == self._DEBOUNCE:
            return raw
        return self.state  # hold current until debounce confirms

    # -----------------------------------------------------------------------
    # Preprocessing for OCR
    # -----------------------------------------------------------------------
    @staticmethod
    def _preprocess(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # 3x upscale — EasyOCR accuracy improves significantly above 60px height
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # CLAHE: local contrast boost for semi-transparent HUD elements
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        # Otsu binarisation: adaptive threshold without manual tuning
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # -----------------------------------------------------------------------
    # OCR — runs on GPU via EasyOCR
    # -----------------------------------------------------------------------
    def _run_ocr(self, bgr: np.ndarray):
        img = self._preprocess(bgr)
        # allowlist: digits, colon, dot, space, "Stage" letters
        results = self.reader.readtext(
            img,
            allowlist='0123456789:. Stage',
            detail=0,          # return strings only, skip bounding boxes
            paragraph=False,
        )
        raw = " ".join(results).strip()
        logging.debug(f"OCR raw: {raw!r}")

        # -- Timer: MM:SS or MM.SS ----------------------------------------
        time_match = re.search(r'(\d{1,2})[:\.](\d{2})', raw)
        if time_match:
            mm, ss = int(time_match.group(1)), int(time_match.group(2))
            # Sanity check: PUBG timers are 00:00-30:00
            if mm <= 30 and ss <= 59:
                self.current_time_str = f"{mm:02d}:{ss:02d}"
                total = mm * 60 + ss
                self.end_time = time.time() + total
                logging.info(f"Timer: {self.current_time_str}  ({total}s remaining)")
            else:
                logging.warning(f"OCR gave implausible time {mm}:{ss} — ignored")

        # -- Stage ---------------------------------------------------------
        stage_match = re.search(r'[Ss]tage\s*(\d)', raw)
        if stage_match:
            self.current_stage = stage_match.group(1)
            logging.info(f"Stage: {self.current_stage}")

    # -----------------------------------------------------------------------
    # Frame-diff guard
    # -----------------------------------------------------------------------
    def _frame_changed(self, gray: np.ndarray) -> bool:
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray.copy()
            return True
        diff = float(np.mean(np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))))
        self._prev_gray = gray.copy()
        return diff > FRAME_DIFF_THRESHOLD

    # -----------------------------------------------------------------------
    # Main frame processing
    # -----------------------------------------------------------------------
    def process_frame(self, sct: mss.mss):
        if not self.region and not self.find_window():
            return

        shot = sct.grab(self.region)
        bgr  = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        raw_state       = self._detect_state(bgr)
        confirmed_state = self._debounced_state(raw_state)

        state_flipped = confirmed_state != self.state
        timer_expired = time.time() > self.end_time
        frame_changed = self._frame_changed(gray)

        if state_flipped or (timer_expired and frame_changed):
            if state_flipped:
                logging.info(f"State  {self.state} -> {confirmed_state}")
                self.state = confirmed_state
            self._run_ocr(bgr)

    # -----------------------------------------------------------------------
    # Run loop
    # -----------------------------------------------------------------------
    def run(self):
        logging.info("PUBG Zone Tracker running — Ctrl+C to stop.")
        with mss.mss() as sct:
            while True:
                try:
                    self.process_frame(sct)

                    remaining = int(self.end_time - time.time())
                    if remaining > 0:
                        if remaining % 15 == 0:
                            logging.info(f"Countdown  {remaining//60:02d}:{remaining%60:02d}  [{self.state}]")
                        time.sleep(2)    # color-check every 2s while counting down
                    else:
                        time.sleep(1)   # idle poll

                except KeyboardInterrupt:
                    logging.info("Stopped by user.")
                    break
                except Exception as exc:
                    logging.error(f"Loop error: {exc}")
                    self.region = None  # re-lookup window on next iteration
                    time.sleep(5)


if __name__ == "__main__":
    tracker = PUBGTracker()
    tracker.run()
