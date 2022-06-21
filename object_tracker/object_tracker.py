import cv2
import numpy as np

def draw_box(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2, 1)
    cv2.putText(img, "Tracking", (50,750), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)    
    return img

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracker = cv2.TrackerMOSSE_create()
# tracker = cv2.TrackerCSRT_create()
_, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
tracker.init(img, bbox)

while True:
    timer = cv2.getTickCount()
    _, img = cap.read()

    success, bbox = tracker.update(img)
    print(bbox)

    if success:
        draw_box(img, bbox)
    else:
        cv2.putText(img, "Lost", (50,750), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)    


    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("Video", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
