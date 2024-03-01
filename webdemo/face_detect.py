import cv2


class FaceDetector:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect(self, image) -> tuple[cv2.typing.Rect]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        return faces
