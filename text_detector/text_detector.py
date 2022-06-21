import cv2
import pytesseract

img = cv2.imread("text2.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(pytesseract.image_to_string(img))

rows, cols, _ = img.shape
boxes = pytesseract.image_to_data(img)
# print(boxes)

conf = 0
count = 0

for x, box in enumerate(boxes.splitlines()):
    if x != 0:
        box = box.split()
        print(box)
        if len(box) == 12:
            conf += float(box[10])
            count += 1
            x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
            cv2.putText(img, box[11], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)


cv2.imshow("Results", img)
cv2.waitKey(0)
