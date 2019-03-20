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
