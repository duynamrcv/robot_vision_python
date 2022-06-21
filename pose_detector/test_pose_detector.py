import cv2
import time
import pose_detector as pd


def main():
    cap = cv2.VideoCapture(0)
    
    detector = pd.poseDetector()

    p_time = 0
    while True:
        _, img = cap.read()

        img = detector.findPose(img)
        lm_list = detector.findPosition(img)
        print(lm_list)

        # if len(lm_list) != 0:
        #     cv2.circle(img, lm_list[14][1], lm_list[14][2], 10, (0,0,255), cv2.FILLED)
    
        # Compute the FPS of video
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (70,80), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        # Display video
        cv2.imshow("video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()