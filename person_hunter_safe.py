import os
import time
import cv2
from ultralytics import YOLO
from djitellopy import Tello

def clamp(x, lo, hi):
    return max(lo, min(hi, int(x)))

# --------- SAFETY / TUNING ----------
CONF_MIN = 0.45

PERSON_TARGET_AREA = 0.040
PERSON_TOO_CLOSE   = 0.080

MAX_FB  = 12
MAX_YAW = 20
MAX_UD  = 0

YAW_GAIN = 30
FB_GAIN  = 500

SEARCH_YAW = 12
SEARCH_TIMEOUT = 15.0
PERSON_FRAMES_TO_LOCK = 5

HOLD_TIME_SEC = 2.5
BACKOFF_TIME_SEC = 2.0
UP_AFTER_TAKEOFF_CM = 40

STREAM_URL = "udp://0.0.0.0:11111"

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

        print("\nARMING (console):")
        print("  Press ENTER to TAKEOFF and start demo")
        print("  Type 'q' + ENTER to quit\n")
        s = input().strip().lower()
        if s == "q":
            print("Quitting.")
            return

        # TAKEOFF
        print("TAKEOFF...")
        tello.takeoff()
        time.sleep(1.0)
        airborne = True

        # Stabilize altitude
        try:
            tello.move_up(UP_AFTER_TAKEOFF_CM)
        except Exception:
            pass
        time.sleep(1.0)

        tello.send_rc_control(0, 0, 0, 0)

        phase = "search"      # search -> approach -> signal -> backoff
        search_start = time.time()
        phase_t0 = time.time()

        seen_count = 0
        locked_person = None

        print("Running. Press Q in the video window to LAND+QUIT.")

        while True:
            ret, img = cap.read()
            if not ret or img is None:
                continue

            h, w = img.shape[:2]
            frame_area = float(w * h)

            # Window quit
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                print("Quit requested (window).")
                break

            # YOLO detect
            res = model(img, verbose=False)[0]
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
            else:
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

                if phase == "approach":
                    if area_frac >= PERSON_TOO_CLOSE:
                        phase = "backoff"
                        phase_t0 = time.time()
                        tello.send_rc_control(0, -MAX_FB, 0, 0)
                    elif area_frac < PERSON_TARGET_AREA:
                        fb = clamp((PERSON_TARGET_AREA - area_frac) * FB_GAIN, 0, MAX_FB)
                        # extra safety: don't approach if far off-center
                        if abs(x_err) > 0.35:
                            fb = 0
                        tello.send_rc_control(0, fb, MAX_UD, yaw)
                    else:
                        phase = "signal"
                        phase_t0 = time.time()
                        tello.send_rc_control(0, 0, 0, 0)

                if phase == "signal":
                    yaw_sig = 15 if int(time.time() * 2) % 2 == 0 else -15
                    tello.send_rc_control(0, 0, 0, yaw_sig)
                    if time.time() - phase_t0 > HOLD_TIME_SEC:
                        phase = "backoff"
                        phase_t0 = time.time()

                if phase == "backoff":
                    tello.send_rc_control(0, -MAX_FB, 0, 0)
                    if time.time() - phase_t0 > BACKOFF_TIME_SEC:
                        print("Backoff complete. Landing.")
                        break

                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, f"{phase.upper()} conf={conf:.2f} area={area_frac:.3f}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Person Hunter (SAFE + SEARCH)", img)

        # exit => land
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()

    finally:
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
