import time
import cv2
from ultralytics import YOLO
from tello_base import Tello

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

def main():
    t = Tello()
    airborne = False

    model = YOLO("yolov8n.pt")  # fast; you can move to yolov8s.pt later
    cat_class = "cat"
    person_class = "person"

    try:
        print("SDK:", t.enter_sdk())

        # Stream on (fire and forget)
        t.send_no_wait("streamon")
        time.sleep(1.0)

        cap = cv2.VideoCapture("udp://0.0.0.0:11111", cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Could not open video stream (udp://0.0.0.0:11111).")

        print("Window controls:")
        print("  T = takeoff (arm)")
        print("  Q = land + quit")
        print("  H = hover (rc 0 0 0 0)")

        # Require explicit takeoff key for safety
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imshow("Cat Safe Shadow", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                return
            if key in (ord("t"), ord("T")):
                break

        print("Takeoff...")
        t.send_no_wait("takeoff")
        time.sleep(3.0)
        airborne = True
        t.send_no_wait("rc 0 0 0 0")

        last_seen_cat = 0.0
        last_cmd_time = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            frame_area = float(w * h)

            res = model(frame, verbose=False)[0]
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

            # Keyboard controls via OpenCV window
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Quit requested")
                break
            if key in (ord("h"), ord("H")):
                t.send_no_wait("rc 0 0 0 0")

            # 1) PERSON OVERRIDE: stop/back away if any person appears
            if best_person is not None:
                _, conf, cx, cy, area_frac, bb = best_person
                x_err = (cx - w/2) / (w/2)
                yaw = clamp(-x_err * YAW_GAIN, -MAX_YAW, MAX_YAW)
                fb = -MAX_FB  # back away slowly but decisively
                if area_frac > PERSON_TOO_CLOSE_AREA_FRAC:
                    fb = -MAX_FB  # keep max backoff
                t.send_no_wait(f"rc 0 {fb} 0 {yaw}")
                last_cmd_time = time.time()

                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"AVOID PERSON {conf:.2f}", (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # 2) CAT TRACKING: shadow, don’t chase
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

                t.send_no_wait(f"rc 0 {fb} {ud} {yaw}")
                last_cmd_time = time.time()

                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"CAT {conf:.2f} area={area_frac:.3f}", (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Draw “too close” warning
                if area_frac >= CAT_TOO_CLOSE_AREA_FRAC:
                    cv2.putText(frame, "TOO CLOSE: BACKING OFF", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # 3) No target: hover (do not wander)
            else:
                # if cat lost recently, just hover and slowly yaw scan (optional)
                if time.time() - last_seen_cat < 2.0:
                    t.send_no_wait("rc 0 0 0 0")
                else:
                    # Very slow scan to reacquire (or comment out to fully hover)
                    t.send_no_wait("rc 0 0 0 10")
                last_cmd_time = time.time()

            cv2.imshow("Cat Safe Shadow", frame)

        # Land on exit
        t.safe_land()

    finally:
        try:
            t.send_no_wait("rc 0 0 0 0")
            t.send_no_wait("streamoff")
        except Exception:
            pass
        try:
            if airborne:
                t.safe_land()
            t.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
