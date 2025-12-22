import os
import time
import cv2
import threading
from ultralytics import YOLO
from djitellopy import Tello


class FrameGrabber:
    """Threaded frame grabber - always has the latest frame ready."""
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False

def clamp(x, lo, hi):
    return max(lo, min(hi, int(x)))

# --------- SAFETY / TUNING ----------
CONF_MIN = 0.45

PERSON_TARGET_AREA = 0.65    # approach until person is 55% of frame (arm's distance)
PERSON_TOO_CLOSE   = 0.75    # back off only if person is 75% of frame

YOLO_IMG_SIZE = 320          # smaller = faster inference (default 640)

MAX_FB  = 20                 # faster forward/back (was 12)
MAX_YAW = 25
MAX_UD  = 0

YAW_GAIN = 35
FB_GAIN  = 400               # approach gain

SEARCH_YAW = 12
SEARCH_TIMEOUT = 15.0
PERSON_FRAMES_TO_LOCK = 5

HOLD_TIME_SEC = 2.5
BACKOFF_TIME_SEC = 2.0
UP_AFTER_TAKEOFF_CM = 40

STREAM_URL = "udp://0.0.0.0:11111"

# Logging
import logging
logging.getLogger("djitellopy").setLevel(logging.WARNING)  # Reduce djitellopy spam

def main():
    weights_path = "./yolov8n.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Missing {weights_path}. Put yolov8n.pt in this folder.\n"
            f"Download once when online:\n"
            f'  python -c "from ultralytics import YOLO; YOLO(\'yolov8n.pt\')"'
        )

    tello = Tello()
    airborne = False

    model = YOLO(weights_path)
    PERSON_CLASS = "person"

    try:
        print("Connecting...")
        tello.connect()
        print("Battery:", tello.get_battery())

        # Reset stream cleanly
        try:
            tello.streamoff()
        except Exception:
            pass
        time.sleep(0.5)
        tello.streamon()
        time.sleep(2.0)

        # Open stream with OpenCV
        print(f"Opening video at: {STREAM_URL}")
        cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video stream")

        # Wait for first frame
        t0 = time.time()
        while time.time() - t0 < 10.0:
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Got first frame!")
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Failed to grab first frame within 10s")

        # Start threaded frame grabber for minimal lag
        grabber = FrameGrabber(cap)

        # Create window early so user can see video before takeoff
        cv2.namedWindow("Person Hunter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Person Hunter", 960, 720)

        print("\nARMING: Press T in the video window to TAKEOFF, Q to quit\n")
        print("Warming up YOLO...")

        # Show video while waiting for takeoff command (run YOLO to warm up)
        warmup_frames = 0
        yolo_ready = False
        while True:
            frame = grabber.get_frame()
            if frame is not None:
                # Run YOLO to keep it warmed up and show detections
                h, w = frame.shape[:2]
                res = model(frame, imgsz=YOLO_IMG_SIZE, verbose=False)[0]
                warmup_frames += 1
                if warmup_frames == 10 and not yolo_ready:
                    print("YOLO warmed up! Ready to fly.")
                    yolo_ready = True

                for box in res.boxes:
                    if float(box.conf[0]) >= CONF_MIN:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names.get(int(box.cls[0]), "")
                        if label == "person":
                            area = ((x2-x1)*(y2-y1)) / (w*h)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"person area={area:.2f}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                status = "READY - Press T" if yolo_ready else f"Warming up YOLO ({warmup_frames}/10)..."
                cv2.putText(frame, status, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if yolo_ready else (0, 255, 255), 2)
                cv2.imshow("Person Hunter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("t"), ord("T")):
                if yolo_ready:
                    break
                else:
                    print("Wait for YOLO to warm up!")
            if key in (ord("q"), ord("Q")):
                print("Quitting.")
                grabber.stop()
                return

        # TAKEOFF
        print("TAKEOFF...")
        print("  [1] Sending takeoff command...")
        tello.takeoff()
        print("  [2] Takeoff complete, waiting 1s...")
        time.sleep(1.0)
        airborne = True
        print("  [3] Airborne = True")

        # Stabilize altitude
        print("  [4] Moving up...")
        try:
            tello.move_up(UP_AFTER_TAKEOFF_CM)
            print("  [5] Move up complete")
        except Exception as e:
            print(f"  [5] Move up failed: {e}")
        time.sleep(1.0)
        print("  [6] Stabilized")

        tello.send_rc_control(0, 0, 0, 0)
        print("  [7] RC control zeroed")

        # Flush buffered frames after blocking commands
        print("Syncing video...")
        for i in range(30):
            grabber.get_frame()
            time.sleep(0.03)
        print("Video synced!")

        phase = "search"      # search -> approach -> signal -> backoff
        search_start = time.time()
        phase_t0 = time.time()

        seen_count = 0
        locked_person = None
        last_phase = None
        frame_count = 0

        print("Running. Press Q in the video window to LAND+QUIT.")
        print("-" * 50)
        print("  [8] Entering main loop...")

        while True:
            # Get latest frame from threaded grabber (minimal lag)
            img = grabber.get_frame()
            if img is None:
                continue

            frame_count += 1
            if frame_count <= 3:
                print(f"  [9] Processing frame {frame_count}...")
            h, w = img.shape[:2]
            frame_area = float(w * h)

            # YOLO detect (smaller input = faster)
            res = model(img, imgsz=YOLO_IMG_SIZE, verbose=False)[0]
            names = model.names

            best_person = None
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONF_MIN:
                    continue
                label = names.get(cls_id, "")
                if label != PERSON_CLASS:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                area_frac = (bw * bh) / frame_area
                cx = (x1 + x2) / 2.0

                best_person = (conf, cx, area_frac, (x1, y1, x2, y2))
                break

            # Persistence lock
            if best_person is not None:
                seen_count += 1
                if seen_count >= PERSON_FRAMES_TO_LOCK:
                    locked_person = best_person
                    if seen_count == PERSON_FRAMES_TO_LOCK:
                        conf, cx, area_frac, _ = best_person
                        print(f"[Frame {frame_count}] LOCKED PERSON: conf={conf:.2f} area={area_frac:.3f}")
            else:
                if locked_person is not None:
                    print(f"[Frame {frame_count}] LOST PERSON")
                seen_count = 0
                locked_person = None

            # Behavior
            if locked_person is None:
                # SEARCH yaw-only
                if time.time() - search_start > SEARCH_TIMEOUT:
                    print("No person found in time. Landing.")
                    break
                tello.send_rc_control(0, 0, 0, SEARCH_YAW)
                remaining = int(SEARCH_TIMEOUT - (time.time() - search_start))
                cv2.putText(img, f"SEARCH yaw-only ({remaining}s)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            else:
                conf, cx, area_frac, bb = locked_person
                x_err = (cx - w/2) / (w/2)
                yaw = clamp(x_err * YAW_GAIN, -MAX_YAW, MAX_YAW)

                if phase == "search":
                    phase = "approach"
                    phase_t0 = time.time()
                    print(f"[Frame {frame_count}] PHASE: search -> approach (area={area_frac:.3f})")

                if phase == "approach":
                    if area_frac >= PERSON_TOO_CLOSE:
                        phase = "backoff"
                        phase_t0 = time.time()
                        print(f"[Frame {frame_count}] PHASE: approach -> backoff (TOO CLOSE area={area_frac:.3f} >= {PERSON_TOO_CLOSE})")
                        tello.send_rc_control(0, -MAX_FB, 0, 0)
                    elif area_frac < PERSON_TARGET_AREA:
                        fb = clamp((PERSON_TARGET_AREA - area_frac) * FB_GAIN, 0, MAX_FB)
                        # Slow down approach if far off-center (but don't stop)
                        if abs(x_err) > 0.4:
                            fb = max(5, fb // 2)
                        tello.send_rc_control(0, fb, MAX_UD, yaw)
                        # Log approach progress periodically
                        if frame_count % 30 == 0:
                            print(f"[Frame {frame_count}] APPROACH: area={area_frac:.3f} target={PERSON_TARGET_AREA} fb={fb}")
                    else:
                        phase = "signal"
                        phase_t0 = time.time()
                        print(f"[Frame {frame_count}] PHASE: approach -> signal (IN RANGE area={area_frac:.3f})")
                        tello.send_rc_control(0, 0, 0, 0)

                if phase == "signal":
                    yaw_sig = 15 if int(time.time() * 2) % 2 == 0 else -15
                    tello.send_rc_control(0, 0, 0, yaw_sig)
                    if time.time() - phase_t0 > HOLD_TIME_SEC:
                        phase = "backoff"
                        phase_t0 = time.time()
                        print(f"[Frame {frame_count}] PHASE: signal -> backoff (signal complete)")

                if phase == "backoff":
                    tello.send_rc_control(0, -MAX_FB, 0, 0)
                    if time.time() - phase_t0 > BACKOFF_TIME_SEC:
                        print("Backoff complete. Landing.")
                        break

                # Log phase changes
                if phase != last_phase:
                    last_phase = phase

                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, f"{phase.upper()} conf={conf:.2f} area={area_frac:.3f}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Show thresholds on screen
                status_color = (0, 255, 0) if area_frac < PERSON_TARGET_AREA else (0, 255, 255) if area_frac < PERSON_TOO_CLOSE else (0, 0, 255)
                cv2.putText(img, f"Target: {PERSON_TARGET_AREA:.3f} | TooClose: {PERSON_TOO_CLOSE:.3f}",
                            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

            # Show frame and check for quit
            cv2.imshow("Person Hunter", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                print("Quit requested (window).")
                break

        # exit => land
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()

    finally:
        try:
            grabber.stop()
        except Exception:
            pass
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.streamoff()
        except Exception:
            pass
        try:
            if airborne:
                tello.land()
            tello.end()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
