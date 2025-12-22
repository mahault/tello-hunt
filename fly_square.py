import time
import keyboard
from djitellopy import Tello

def main():
    tello = Tello()
    airborne = False

    try:
        print("Connecting...")
        tello.connect()
        print("Battery:", tello.get_battery(), "%")

        print("Takeoff")
        tello.takeoff()
        time.sleep(1.0)
        airborne = True

        # Square flight
        for i in range(4):

            # KILL SWITCH CHECK
            if keyboard.is_pressed("q"):
                print("EMERGENCY LAND")
                tello.land()
                return

            print(f"Forward {i+1}")
            tello.move_forward(50)
            time.sleep(0.5)

            if keyboard.is_pressed("q"):
                print("EMERGENCY LAND")
                tello.land()
                return

            print(f"CW {i+1}")
            tello.rotate_clockwise(90)
            time.sleep(0.5)

        print("Square complete, landing...")
        tello.land()
        airborne = False

    finally:
        if airborne:
            print("Landing (failsafe)")
            tello.land()
            time.sleep(1.0)
        tello.end()
        print("Done")

if __name__ == "__main__":
    main()
