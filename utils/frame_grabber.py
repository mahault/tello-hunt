"""
Threaded frame grabber for minimal video latency.

Provides a FrameGrabber class that continuously reads frames from
a video source in a background thread, ensuring get_frame() always
returns the most recent frame without blocking.
"""

import threading
from typing import Optional
import numpy as np


class FrameGrabber:
    """
    Threaded frame grabber - always has the latest frame ready.

    Runs a background thread that continuously reads from the video
    capture, so get_frame() always returns the most recent frame
    without blocking. This minimizes latency for safety decisions.

    Usage:
        cap = cv2.VideoCapture(stream_url)
        grabber = FrameGrabber(cap)

        while running:
            frame = grabber.get_frame()
            if frame is not None:
                # Process frame
                pass

        grabber.stop()
    """

    def __init__(self, cap):
        """
        Initialize frame grabber with a cv2.VideoCapture.

        Args:
            cap: OpenCV VideoCapture instance (already opened)
        """
        self.cap = cap
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = True
        self._frame_count = 0
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self) -> None:
        """Background loop that continuously grabs frames."""
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self._frame_count += 1

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame (thread-safe copy).

        Returns:
            Latest frame as numpy array (BGR format), or None if no frame available
        """
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_frame_count(self) -> int:
        """
        Get the number of frames grabbed since start.

        Returns:
            Total frame count
        """
        with self.lock:
            return self._frame_count

    def stop(self) -> None:
        """Stop the background grabber thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def __del__(self):
        """Ensure thread is stopped on cleanup."""
        self.stop()


def clamp(x: float, lo: float, hi: float) -> int:
    """
    Clamp a value to a range and convert to int.

    Args:
        x: Value to clamp
        lo: Minimum value
        hi: Maximum value

    Returns:
        Clamped integer value
    """
    return max(lo, min(hi, int(x)))
