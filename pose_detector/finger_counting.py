import cv2
import os
import time
import hand_detector as hd

w_cam, h_cam = 480, 640
cap = cv2.VideoCapture(0)

folder_path = "finger_images"
finger_list = os.listdir(folder_path)
overlay_list = []
for im_path in finger_list:
    image = cv2.imread(f'{folder_path}/{im_path}')
    overlay_list.append(image)

detector = hd.handDetector(detectionCon=0.75)
tip_idxs = [4, 8, 12, 16, 20]

p_time = 0

while True:
    _, img = cap.read()
    cv2.imshow("Video", img)

    img = detector.findHands(img)

    lm_list = detector.findPosition(img, draw=False)
    # print(lm_list)

    if len(lm_list) != 0:
        fingers = []

        # Thumb
        if lm_list[tip_idxs[0]][1] > lm_list[tip_idxs[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for i in range(1, len(tip_idxs)):
            if lm_list[tip_idxs[i]][2] < lm_list[tip_idxs[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        total_fingers = fingers.count(1)
        print(total_fingers)
        cv2.putText(img, str(total_fingers), (40,300), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 10)


        h, w, c = overlay_list[total_fingers].shape
        img[0:h, 0:w] = overlay_list[total_fingers]

    # FPS
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    # Display video
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()