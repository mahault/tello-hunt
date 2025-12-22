import time
import cv2
from djitellopy import Tello

# OpenCV FFmpeg backend URL for Tello
STREAM_URL = "udp://0.0.0.0:11111"

def main():
    tello = Tello()

    print("Connecting...")
    tello.connect()
    print("Battery:", tello.get_battery())

    # Reset stream state
    try:
        tello.streamoff()
    except Exception:
        pass

    print("Starting stream...")
    tello.streamon()
    time.sleep(2.0)

    # Open stream with OpenCV (uses FFmpeg backend)
    print(f"Opening video at: {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Failed to open video stream with OpenCV")
        tello.streamoff()
        tello.end()
        return

    print("Video window should open. Press Q to quit.")

    # Warm-up: wait for first valid frame
    t0 = time.time()
    while time.time() - t0 < 10.0:
        ret, frame = cap.read()
        if ret and frame is not None:
            print("Got first frame!")
            break
        time.sleep(0.05)
    else:
        print("Failed to grab first frame within 10s.")
        cap.release()
        tello.streamoff()
        tello.end()
        return

    # Display loop
    while True:
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imshow("Tello Stream (OpenCV)", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()
    cap.release()
    tello.streamoff()
    tello.end()
    print("Done.")

if __name__ == "__main__":
    main()
