import cv2
import numpy as np
from pyzbar.pyzbar import decode

# img = cv2.imread("barcode.jpg")
cap = cv2.VideoCapture(-1)

while True:
    success, img = cap.read()
    if not success:
        continue

    barcodes = decode(img)
    if len(barcodes) != 0:
        for barcode in barcodes:
            # print(barcode.rect)
            data = barcode.data.decode('utf-8')
            print(data)

            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img, [pts], True, (255,0,0), 5)

            pts2 = barcode.rect
            cv2.putText(img, data, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    
    cv2.imshow("Video", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break