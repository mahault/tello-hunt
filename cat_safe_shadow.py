import time
import cv2
import logging
import threading
from ultralytics import YOLO
from djitellopy import Tello

logging.getLogger("djitellopy").setLevel(logging.WARNING)  # Reduce log spam


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
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
            except:
                break

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        time.sleep(0.1)

def clamp(x, lo, hi):
    return max(lo, min(hi, int(x)))

# ---- Safety tuning (start conservative) ----
CONF_MIN = 0.45

CAT_TARGET_AREA_FRAC = 0.040     # "comfortable" standoff
CAT_TOO_CLOSE_AREA_FRAC = 0.090  # back off early
CAT_TOO_FAR_AREA_FRAC = 0.015    # optional: slow approach if really small

# Hard speed limits (pets nearby => keep slow)
MAX_YAW = 25     # -100..100 (keep small)
MAX_FB  = 18     # forward/back (slow)
MAX_UD  = 12     # altitude changes (very small)

# Gains
YAW_GAIN = 30
FB_GAIN  = 600   # area error is tiny; scaled up, still clamped by MAX_FB
UD_GAIN  = 20    # keep small or set to 0 to disable vertical motion

# Person safety
PERSON_TOO_CLOSE_AREA_FRAC = 0.06

YOLO_IMG_SIZE = 320  # smaller = faster inference

STREAM_URL = "udp://0.0.0.0:11111"

def main():
    tello = Tello()
    airborne = False

    model = YOLO("yolov8n.pt")  # fast; you can move to yolov8s.pt later
    cat_class = "cat"
    person_class = "person"

    try:
        print("Connecting...")
        tello.connect()
        print("Battery:", tello.get_battery())

        # Stream on
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.streamon()
        time.sleep(2.0)

        cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream ({STREAM_URL}).")

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

        # Start threaded frame grabber
        grabber = FrameGrabber(cap)

        # Create window early
        cv2.namedWindow("Cat Safe Shadow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cat Safe Shadow", 960, 720)

        print("\nARMING: Press T in the video window to TAKEOFF, Q to quit")
        print("  H = hover (during flight)")
        print("Warming up YOLO...")

        # Show video while waiting for takeoff command (run YOLO to warm up)
        warmup_frames = 0
        yolo_ready = False
        while True:
            frame = grabber.get_frame()
            if frame is not None:
                h, w = frame.shape[:2]
                res = model(frame, imgsz=YOLO_IMG_SIZE, verbose=False)[0]
                warmup_frames += 1
                if warmup_frames == 10 and not yolo_ready:
                    print("YOLO warmed up! Ready to fly.")
                    yolo_ready = True

                # Show detections during warmup
                for box in res.boxes:
                    if float(box.conf[0]) >= CONF_MIN:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names.get(int(box.cls[0]), "")
                        if label in ("cat", "person"):
                            area = ((x2-x1)*(y2-y1)) / (w*h)
                            color = (0, 255, 0) if label == "cat" else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{label} area={area:.2f}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                status = "READY - Press T" if yolo_ready else f"Warming up YOLO ({warmup_frames}/10)..."
                cv2.putText(frame, status, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if yolo_ready else (0, 255, 255), 2)
                cv2.imshow("Cat Safe Shadow", frame)

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

        print("Takeoff...")
        tello.takeoff()
        time.sleep(1.0)
        airborne = True
        tello.send_rc_control(0, 0, 0, 0)

        # Flush buffered frames after takeoff
        print("Syncing video...")
        for _ in range(30):
            grabber.get_frame()
            time.sleep(0.03)
        print("Video synced!")

        last_seen_cat = 0.0

        print("Running. Press Q to land+quit, H to hover.")
        print("-" * 50)

        while True:
            frame = grabber.get_frame()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            frame_area = float(w * h)

            res = model(frame, imgsz=YOLO_IMG_SIZE, verbose=False)[0]
            names = model.names

            best_cat = None
            best_person = None

            # pick largest cat/person by area (closest proxy)
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONF_MIN:
                    continue

                label = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                area_frac = (bw * bh) / frame_area
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                item = (label, conf, cx, cy, area_frac, (x1, y1, x2, y2))

                if label == person_class:
                    if (best_person is None) or (area_frac > best_person[4]):
                        best_person = item
                elif label == cat_class:
                    if (best_cat is None) or (area_frac > best_cat[4]):
                        best_cat = item

            # 1) PERSON OVERRIDE: stop/back away if any person appears
            if best_person is not None:
                _, conf, cx, cy, area_frac, bb = best_person
                x_err = (cx - w/2) / (w/2)
                yaw = clamp(-x_err * YAW_GAIN, -MAX_YAW, MAX_YAW)
                fb = -MAX_FB  # back away slowly but decisively
                if area_frac > PERSON_TOO_CLOSE_AREA_FRAC:
                    fb = -MAX_FB  # keep max backoff
                tello.send_rc_control(0, fb, 0, yaw)

                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"AVOID PERSON {conf:.2f}", (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # 2) CAT TRACKING: shadow, don't chase
            elif best_cat is not None:
                label, conf, cx, cy, area_frac, bb = best_cat
                last_seen_cat = time.time()

                # Centering errors
                x_err = (cx - w/2) / (w/2)  # -1..1
                y_err = (cy - h/2) / (h/2)  # -1..1

                # Distance control using area fractions
                if area_frac >= CAT_TOO_CLOSE_AREA_FRAC:
                    # Too close => back away + hover attitude stable
                    fb = -MAX_FB
                elif area_frac <= CAT_TOO_FAR_AREA_FRAC:
                    # Far => VERY gentle approach
                    fb = clamp((CAT_TARGET_AREA_FRAC - area_frac) * FB_GAIN, 0, MAX_FB)
                else:
                    # In band => no forward motion
                    fb = 0

                yaw = clamp(x_err * YAW_GAIN, -MAX_YAW, MAX_YAW)

                # Keep altitude mostly stable (or set ud=0 to disable)
                ud = clamp(-y_err * UD_GAIN, -MAX_UD, MAX_UD)

                # Extra safety: if moving forward at all, reduce yaw to avoid arcing
                if fb > 0:
                    yaw = clamp(yaw, -15, 15)

                tello.send_rc_control(0, fb, ud, yaw)

                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"CAT {conf:.2f} area={area_frac:.3f}", (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Draw "too close" warning
                if area_frac >= CAT_TOO_CLOSE_AREA_FRAC:
                    cv2.putText(frame, "TOO CLOSE: BACKING OFF", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # 3) No target: hover (do not wander)
            else:
                # if cat lost recently, just hover and slowly yaw scan (optional)
                if time.time() - last_seen_cat < 2.0:
                    tello.send_rc_control(0, 0, 0, 0)
                else:
                    # Very slow scan to reacquire (or comment out to fully hover)
                    tello.send_rc_control(0, 0, 0, 10)

            # Show frame and check for keyboard
            cv2.imshow("Cat Safe Shadow", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Quit requested")
                break
            if key in (ord("h"), ord("H")):
                tello.send_rc_control(0, 0, 0, 0)

        # Land on exit
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
