import time
import keyboard
from tello_base import Tello

def main():
    t = Tello()
    airborne = False

    try:
        print("Entering SDK mode:", t.enter_sdk())

        print("Takeoff")
        t.send_no_wait("takeoff")
        time.sleep(3)
        airborne = True

        # Square flight
        for i in range(4):

            # 🔴 KILL SWITCH CHECK (runs continuously)
            if keyboard.is_pressed("q"):
                print("EMERGENCY LAND")
                t.safe_land()
                return   # exit main(), then finally{} will run

            print(f"Forward {i+1}")
            t.send_expect("forward 50", timeout_s=6.0, retries=2)
            time.sleep(1.0)

            if keyboard.is_pressed("q"):
                print("EMERGENCY LAND")
                t.safe_land()
                return

            print(f"CW {i+1}")
            t.send_expect("cw 90", timeout_s=6.0, retries=2)
            time.sleep(1.0)

    finally:
        if airborne:
            print("Landing (failsafe)")
            t.safe_land()
            time.sleep(1.0)
        t.close()
        print("Done")

if __name__ == "__main__":
    main()
