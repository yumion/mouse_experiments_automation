import pandas as pd
from glob import glob
import os
import shutil


# 教師ラベル
df = pd.read_csv('raw_data/200170616_BT_NS_038/digtime_NS38.csv')
start_time = df['start'].to_list()
end_time = df['end'].to_list()

# 入力データ
frames_file = glob('raw_data/200170616_BT_NS_038/frames/*')
frames_file = sorted(frames_file)

def time_to_sec(time):
    '''時刻を秒数へ変換'''
    if ':' in time: # excel表記
        min, sec = time.split(':')
    elif '-' in time: # ファイル名表記
        min, sec = time.split('-')
    min, sec = float(min), float(sec)
    sec = min*60 + sec
    return sec

# 秒へ変換
start_time = [time_to_sec(t) for t in start_time] # 教師ラベル
end_time = [time_to_sec(t) for t in end_time]
frames_time = [os.path.basename(f) for f in frames_file] # ファイル名取得
frames_time = [os.path.splitext(f)[0] for f in frames_time] # 拡張子を取り除く
frames_time = [time_to_sec(f) for f in frames_time] # 入力データ

# 穴掘り時間のフレームをラベリング
for i, ft in enumerate(frames_time): # まずフレームを固定して
    for st, et in zip(start_time, end_time): # 穴掘り時間に該当するか探す
        if st <= ft and et >= ft: # 穴掘りしている時間
            cls_dir = os.path.join('dataset/1', os.path.basename(frames_file[i]))
        else: # その他の時間
            cls_dir = os.path.join('dataset/0', os.path.basename(frames_file[i]))
        shutil.copyfile(frames_file[i], cls_dir) # フレームごとに保存するためにこの位置で


# ０フォルダにも１フォルダの画像がコピーされたので、重複を消す
cls_0 = sorted(glob('dataset/0/*'))
cls_0_name = [os.path.basename(f) for f in cls_0]
cls_1 = sorted(glob('dataset/1/*'))
cls_1_name = [os.path.basename(f) for f in cls_1]
for i in range(len(cls_1)):
    if cls_1_name[i] in cls_0_name:
        rm_file = cls_0[cls_0_name.index(cls_1_name[i])]
        # print(rm_file)
        os.remove(rm_file)
