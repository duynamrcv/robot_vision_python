import cv2
import mediapipe as mp

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upBody, self.smooth, 
                                self.detectionCon, self.trackingCon)
    
    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(idx, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lm_list
