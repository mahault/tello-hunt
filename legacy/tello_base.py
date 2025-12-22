import socket
import threading
import time
from typing import Optional

class Tello:
    """
    Robust minimal Tello/Tello Talent UDP SDK client for Direct Connection mode.

    Key changes:
    - close() does NOT auto-land unless auto_land=True
    - safe_land() available for flight scripts
    - emergency() available as last resort
    """

    def __init__(
        self,
        tello_ip: str = "192.168.10.1",
        cmd_port: int = 8889,
        local_port: int = 8889,
        verbose: bool = True,
        auto_land: bool = False,
    ):
        self.tello_addr = (tello_ip, cmd_port)
        self.verbose = verbose
        self.auto_land = auto_land

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", local_port))
        self.sock.settimeout(0.5)

        self._last_response: Optional[str] = None
        self._running = True

        self._listen_thread = threading.Thread(target=self._listen, daemon=True)
        self._listen_thread.start()

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _listen(self):
        while self._running:
            try:
                data, _ = self.sock.recvfrom(2048)
                resp = data.decode("utf-8", errors="ignore").strip()
                self._last_response = resp
                self._log(f"[RECV] {resp}")
            except (socket.timeout, TimeoutError):
                continue
            except Exception:
                continue

    def send_no_wait(self, cmd: str):
        self._log(f"[SEND] {cmd}")
        try:
            self.sock.sendto(cmd.encode("utf-8"), self.tello_addr)
        except Exception as e:
            self._log(f"[SEND-ERR] {cmd} -> {e}")

    def send_expect(self, cmd: str, timeout_s: float = 5.0, retries: int = 2) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            self._last_response = None
            self._log(f"[SEND] {cmd} (attempt {attempt+1}/{retries+1})")
            try:
                self.sock.sendto(cmd.encode("utf-8"), self.tello_addr)
            except Exception as e:
                last_err = e
                self._log(f"[SEND-ERR] {cmd} -> {e}")
                time.sleep(0.2)
                continue

            t0 = time.time()
            while time.time() - t0 < timeout_s:
                if self._last_response is not None:
                    return self._last_response
                time.sleep(0.01)

            last_err = TimeoutError(f"No response to '{cmd}' within {timeout_s}s")
            self._log(f"[TIMEOUT] {cmd}")

        raise last_err if last_err else TimeoutError(f"No response to '{cmd}'")

    def enter_sdk(self) -> str:
        return self.send_expect("command", timeout_s=3.0, retries=3)

    def safe_land(self):
        """Best-effort landing sequence."""
        self._log("[SAFE] stop + land")
        try:
            self.send_no_wait("rc 0 0 0 0")
            time.sleep(0.1)
            for _ in range(3):
                self.send_no_wait("land")
                time.sleep(0.25)
        except Exception:
            pass

    def emergency(self):
        """Emergency motor cut (drone may drop)."""
        self._log("[EMERGENCY] motor cut")
        self.send_no_wait("emergency")

    def close(self):
        self._running = False
        if self.auto_land:
            try:
                self.safe_land()
            except Exception:
                pass
        try:
            self.sock.close()
        except Exception:
            pass
