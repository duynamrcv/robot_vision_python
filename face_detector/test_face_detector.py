import cv2
import time
import face_detector as fd

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ptime = time.time()

    detector = fd.FaceDetector()

    while True:
        _, img = cap.read()

        img, bboxs = detector.findFaces(img)

        # Show FPS
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255,0,0), 2)

        cv2.imshow("Video", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break