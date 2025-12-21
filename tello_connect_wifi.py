import socket
import time

# In direct connection mode, Tello/Tello Talent commonly listens here:
DRONE_IP = "192.168.10.2"
CMD_PORT = 8889

SSID = "VIRGIN837"
PASSWORD = "5DF26AFFF261"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", 0))
sock.settimeout(5)

def send(cmd: str) -> str:
    sock.sendto(cmd.encode("utf-8"), (DRONE_IP, CMD_PORT))
    data, _ = sock.recvfrom(1024)
    return data.decode("utf-8", errors="ignore")

print("SDK mode:", send("command"))          # expect "ok"
print("Set router:", send(f"ap {SSID} {PASSWORD}"))  # expect OK + reboot message
print("Waiting for reboot...")
time.sleep(6)
print("Done. Now connect your laptop back to your home Wi-Fi.")
