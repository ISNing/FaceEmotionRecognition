import numpy as np
from cv2.typing import MatLike
from picamera2 import Picamera2


class Camera:
    def capture(self) -> MatLike:
        raise NotImplementedError()

    def start(self):
        pass

    def close(self):
        pass


class PiCamera(Camera):
    picam2 = None
    capture_config = None

    def __init__(self, capture_config=None):
        self.picam2 = Picamera2()
        if not capture_config:
            self.capture_config = self.picam2.create_video_configuration()
        else:
            self.capture_config = capture_config

    def start(self):
        self.picam2.start()

    def capture(self) -> MatLike:
        return np.array(self.picam2.capture_array())

    def close(self):
        self.picam2.close()
