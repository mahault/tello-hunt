import time
import av
import cv2
from tello_base import Tello

STREAM_URL = "udp://0.0.0.0:11111"

def main():
    t = Tello(verbose=True)
    try:
        print("SDK:", t.enter_sdk())

        # Reset stream cleanly
        t.send_no_wait("streamoff")
        time.sleep(0.5)
        t.send_no_wait("streamon")
        time.sleep(1.0)

        print("Opening PyAV stream...")
        container = av.open(STREAM_URL, format="h264", timeout=5)

        print("Video OK. Press Q in the window to quit.")
        last = time.time()

        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="bgr24")
            cv2.imshow("TT Video (PyAV)", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

            # small heartbeat
            if time.time() - last > 2:
                print("...receiving video frames")
                last = time.time()

        cv2.destroyAllWindows()

    finally:
        try:
            t.send_no_wait("streamoff")
        except Exception:
            pass
        t.close()

if __name__ == "__main__":
    main()
