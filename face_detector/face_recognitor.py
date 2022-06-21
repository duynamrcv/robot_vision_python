import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "image_attendance"
images = []
names = []
face_list = os.listdir(path)
print(face_list)

for cl in face_list:
    cur_img = cv2.imread(f'{path}/{cl}')
    images.append(cur_img)
    names.append(os.path.splitext(cl)[0])
print(names)

def findEncodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            dt_str = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dt_str}')

encode_list = findEncodings(images)
print(len(encode_list))
print("Encoding done!")

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = face_recognition.face_locations(img)
    encodes = face_recognition.face_encodings(img, faces)

    for encode, face in zip(encodes, faces):
        matches = face_recognition.compare_faces(encode_list, encode)
        face_dis = face_recognition.face_distance(encode_list, encode)
        print(face_dis)

        match_index = np.argmin(face_dis)
        if matches[match_index]:
            name = names[match_index].upper()
            print(name)
            y1, x2, y2, x1 = face
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-30), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+5, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markAttendance(name)

    cv2.imshow("Video", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break