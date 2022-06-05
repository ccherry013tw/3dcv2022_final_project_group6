
import cv2
import numpy as np
import sys

def createBox(img,points):

    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,[points],(255,255,255))
    img = cv2.bitwise_and(img,mask)
    return mask


def try_lips(image_file_path, landmark_file_path, r, g, b):
    img = cv2.imread(image_file_path)
    imgOriginal = img.copy()

    landmarks = np.load(landmark_file_path)
    myPoints =[]

    # could adjust landmarks to select where you want to color
    for n in range(76, 89):
        x = landmarks[n, 0]
        y = landmarks[n, 1]
        myPoints.append([x,y])

    myPoints = np.array(myPoints, 'int32')
    imgLips = createBox(img,myPoints[:])

    imgColorLips = np.zeros_like(imgLips)
    imgColorLips[:] = int(b), int(g), int(r) # b, g, r
    imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)

    imgColorLips = cv2.bitwise_and(imgLips,imgColorLips)
    imgColorLips = cv2.GaussianBlur(imgColorLips,(7,7),25)
    imgColorLips = cv2.addWeighted(imgOriginal,1,imgColorLips,0.2,0)
    cv2.imwrite('output_lips.jpg', imgColorLips)

def main(image_file_path, landmark_file_path, r, g, b):
    try_lips(image_file_path, landmark_file_path, r, g, b)

# argv0: image file path
# argv1: landmark_file_path
# argv2, 3, 4: r, g, b
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
