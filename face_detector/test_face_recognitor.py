import cv2
import numpy as np
import face_recognition

img_nam = face_recognition.load_image_file("images/DuyNam.jpg")
img_nam = cv2.cvtColor(img_nam, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file("images/Tesla.jpg")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(img_nam)[0]
encode_nam = face_recognition.face_encodings(img_nam)[0]
cv2.rectangle(img_nam, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255,0,0), 2)

face_loc_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_loc_test[3], face_loc_test[0]), (face_loc_test[1], face_loc_test[2]), (255,0,0), 2)

results = face_recognition.compare_faces([encode_nam], encode_test)
face_dis = face_recognition.face_distance([encode_nam], encode_test)
print(results, face_dis)

cv2.imshow("Duy Nam", img_nam)
cv2.imshow("test", img_test)
cv2.waitKey(0)