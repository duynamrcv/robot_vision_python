import cv2
import numpy as np

def getContours(img, canny_thresh=[100, 100], min_area=500, filter=0, show_canny=False, draw=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    canny = cv2.Canny(blur, canny_thresh[0], canny_thresh[1])

    kernel = np.ones((5,5))
    dilate = cv2.dilate(canny, kernel, iterations=3)
    erode = cv2.erode(dilate, kernel, iterations=3)
    if show_canny:
        cv2.imshow("canny", erode)
    
    contours, hiearchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)

            if filter > 0:
                if len(approx) == filter:
                    final_contours.append([len(approx), area, approx, bbox, contour])
            else:
                final_contours.append([len(approx), area, approx, bbox, contour])
    
    final_contours = sorted(final_contours, key=lambda x:x[1], reverse=True)
    if draw:
        for contour in final_contours:
            cv2.drawContours(img, contour[4], -1, [255,0,0], 3)
    
    return img, final_contours

def reorder(points):
    print(points.shape)
    points_new = np.zeros_like(points)
    points = points.reshape((4, 2))
    add = points.sum(1)
    
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]

    return points_new

def warp_image(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (w,h))
    img_warp = img_warp[pad:img_warp.shape[0]-pad, pad:img_warp.shape[1]-pad]

    return img_warp

def find_distance(pts1, pts2):
    return np.sqrt((pts2[0] - pts1[0])**2 + (pts2[1] - pts1[1])**2)