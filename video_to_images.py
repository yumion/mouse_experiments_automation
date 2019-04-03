import cv2
import os

## パスを指定 ##
video_path = 'raw_data/200170616_BT_NS_038/200170616_BT_NS_038.m4v'

cap = cv2.VideoCapture(video_path)
# プロパティ
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 幅
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 高さ
count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 総フレーム数
fps = cap.get(cv2.CAP_PROP_FPS) # fps
print('width:{}, height:{}, count:{}, fps:{}'.format(width, height, count, fps))

# 保存するフォルダを作成
frame_path = os.path.join(video_path.split('.')[0], 'frames')
os.makedirs(frame_path, exist_ok=True)

# フレームごとに画像で保存
num = 0
time_min = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    # 時刻に変換
    total_sec = num / fps
    time_min = int(total_sec / 60)
    time_sec = total_sec - time_min*60
    # 保存
    if ret == True:
        output_path = os.path.join(frame_path, '{0:02}-{1:05.2f}.jpg'.format(time_min, time_sec))
        cv2.imwrite(output_path, frame)
        # print("save {0:02}-{1:05.2f}.jpg".format(time_min, time_sec))
        num += 1
    else:
        break
cap.release()
