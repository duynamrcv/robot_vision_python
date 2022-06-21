import cv2
import numpy as np
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDectectionCon=0.5):
        self.minDectectionCon = minDectectionCon
        self.mp_face = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face.FaceDetection(self.minDectectionCon)

    def findFaces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id, detection)
                # mp_draw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                        int (bboxC.width*iw), int(bboxC.height*ih)

                bboxs.append([id, bbox, detection.score])
                
                if draw:
                    cv2.rectangle(img, bbox, (255,0,255), 2)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-10),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)
        return img, bboxs