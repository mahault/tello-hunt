# import socket

# DRONE_IP = "192.168.1.57"  # <-- IP from your router’s device list
# CMD_PORT = 8889

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind(("", 0))
# sock.settimeout(5)

# def send(cmd: str) -> str:
#     sock.sendto(cmd.encode("utf-8"), (DRONE_IP, CMD_PORT))
#     data, _ = sock.recvfrom(1024)
#     return data.decode("utf-8", errors="ignore")

# print(send("command"))
# print(send("takeoff"))
# print(send("forward 50"))
# print(send("land"))

import socket
import time

DRONE_IP = "192.168.10.1"
DRONE_PORT = 8889

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", 8889))          # bind locally to 8889 to reliably receive replies
sock.settimeout(2)

def send(cmd: str) -> str:
    sock.sendto(cmd.encode("utf-8"), (DRONE_IP, DRONE_PORT))
    data, _ = sock.recvfrom(1024)
    return data.decode("utf-8", errors="ignore")

for attempt in range(5):
    try:
        print("Sending: command")
        print("Reply:", send("command"))
        break
    except TimeoutError:
        print(f"Timeout (attempt {attempt+1}/5). Retrying...")
        time.sleep(0.5)

sock.close()
