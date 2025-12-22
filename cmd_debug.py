import time
import msvcrt
from djitellopy import Tello

def read_key_nonblocking():
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        try:
            return ch.decode("utf-8", errors="ignore").lower()
        except Exception:
            return None
    return None

def main():
    tello = Tello()
    airborne = False

    try:
        print("Connecting...")
        tello.connect()

        # Safety gate
        bat = tello.get_battery()
        print("Battery:", bat, "%")
        if bat < 20:
            print("Battery too low to fly safely. Charge first.")
            return

        print("\nControls (no Enter needed):")
        print("  T = takeoff")
        print("  L = land immediately")
        print("  Q = land + quit")
        print("  E = EMERGENCY motor cut (last resort)")
        print("  Ctrl+C = land + quit\n")

        last_print = time.time()

        while True:
            k = read_key_nonblocking()

            if k == "t" and not airborne:
                print("[CTRL] TAKEOFF")
                tello.takeoff()
                airborne = True
                time.sleep(1.0)
                tello.send_rc_control(0, 0, 0, 0)

            elif k == "l":
                print("[CTRL] LAND")
                tello.land()
                airborne = False

            elif k == "q":
                print("[CTRL] QUIT (land first)")
                if airborne:
                    tello.land()
                break

            elif k == "e":
                print("[CTRL] EMERGENCY")
                tello.emergency()
                break

            # periodic heartbeat
            if time.time() - last_print > 2.0:
                last_print = time.time()
                try:
                    h = tello.get_height()
                    print("[INFO] height:", h, "cm")
                except Exception:
                    pass

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[CTRL] Ctrl+C -> landing")
        if airborne:
            tello.land()

    finally:
        tello.end()
        print("Done.")

if __name__ == "__main__":
    main()
