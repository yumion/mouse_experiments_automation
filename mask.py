import cv2
import matplotlib.pyplot as plt

img_path = 'raw_data/200170616_BT_NS_038/frames/11-27.87.jpg'
img = cv2.imread(img_path)
# グレースケールに変換する。
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 2値化する
_, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
# original
plt.imshow(img)
plt.show()
# gray scale
plt.imshow(gray, 'gray')
plt.show()
# binary scale
plt.imshow(binary, 'gray')
plt.show()


img = cv2.medianBlur(img, 5)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
