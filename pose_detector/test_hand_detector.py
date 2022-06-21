import cv2
import time
import hand_detector as hd

def main():
    cap = cv2.VideoCapture(0)
    detector = hd.handDetector()
    p_time = 0

    while True:
        _, img = cap.read()

        img = detector.findHands(img)
        lm_list = detector.findPosition(img)

        if len(lm_list) != 0:
            print(lm_list)

        # FPS
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        # Display video
        cv2.imshow("video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()