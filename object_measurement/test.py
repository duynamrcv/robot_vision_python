import cv2
import numpy as np
import measurement as me

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    img, contours = me.getContours(img, min_area=500, filter=4, show_canny=True, draw=True)

    if len(contours) != 0:
        biggest = contours[0][2]
        # print(biggest)
        img_warp = me.warp_image(img, biggest, 300, 300)
        # cv2.imshow("Warp", img_warp)

        img_con2, contours2 = me.getContours(img_warp, min_area=100, filter=4,
                                            canny_thresh=[50,50], draw=True)
        if len(contours2) != 0:
            for obj in contours2:
                cv2.polylines(img_con2, [obj[2]], True, (0,255,255), 2)
                npoints = me.reorder(obj[2])
                nH = round((me.find_distance(npoints[0][0], npoints[1][0])/10), 1)
                nW = round((me.find_distance(npoints[0][0], npoints[2][0])/10), 1)

                cv2.arrowedLine(img, (npoints[0][0][0], npoints[0][0][1]), (npoints[1][0][0], npoints[1][0][1]),
                                (255,0,255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img, (npoints[0][0][0], npoints[0][0][1]), (npoints[2][0][0], npoints[2][0][1]),
                                (255,0,255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(img, "{}cm".format(nW), (x+30, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255,0,255), 2)
                cv2.putText(img, "{}cm".format(nH), (x-70, y+h//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255,0,255), 2)

        cv2.imshow("A4", img_con2)

    cv2.imshow("Origin", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
