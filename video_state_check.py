# video_state_check.py
from tello_base import Tello
import time

t = Tello(verbose=True, auto_land=False)
print("SDK:", t.enter_sdk())

for cmd in [
    "streamon",
    "streamon",          # send twice (some firmwares are picky)
    "setfps 15",
    "setresolution low",
    "setbitrate 2",
]:
    try:
        print(cmd, "->", t.send_expect(cmd, timeout_s=2, retries=1))
    except Exception as e:
        print(cmd, "-> no reply", e)

time.sleep(2)
print("Done. Now try video_debug_pyav.py again.")
t.close()
