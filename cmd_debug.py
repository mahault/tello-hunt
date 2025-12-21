import time
from tello_base import Tello

def main():
    t = Tello(verbose=True)
    airborne = False
    try:
        print("SDK:", t.enter_sdk())

        # Optional queries (may or may not reply depending on firmware)
        for q in ["battery?", "time?", "height?", "temp?"]:
            try:
                print(q, "->", t.send_expect(q, timeout_s=2.0, retries=1))
            except Exception as e:
                print(q, "-> (no reply)", repr(e))

        print("\nPress ENTER to TAKEOFF (console). Type 'q' + ENTER to quit without takeoff.")
        s = input().strip().lower()
        if s == "q":
            print("Quitting.")
            return

        print("TAKEOFF now (no-wait).")
        t.send_no_wait("takeoff")
        airborne = True

        # Wait so you can visually confirm
        for i in range(10):
            print(f"...hovering {i+1}/10 (type 'l' + ENTER anytime to land)")
            time.sleep(1)

        print("LAND now.")
        t.safe_land()
        time.sleep(2)

    finally:
        if airborne:
            t.safe_land()
        t.close()
        print("Done.")

if __name__ == "__main__":
    main()
