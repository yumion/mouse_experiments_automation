import pandas as pd
from glob import glob
import os
import shutil


# 教師ラベル
df = pd.read_csv('raw_data/200170616_BT_NS_038/digtime_NS38.csv')
start_time = df['start'].to_list()
end_time = df['end'].to_list()

# 入力データ
frames_file = glob('raw_data/200170616_BT_NS_038/200170616_BT_NS_038/frames/*')
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
with open('annotation_NS38.csv', 'w') as f:
    # f.write('frame,class\n')
    for i, ft in enumerate(frames_time): # まずフレームを固定して
        for st, et in zip(start_time, end_time): # 穴掘り時間に該当するか探してラベリング
            if st <= ft and et >= ft: # 穴掘りしている時間
                cls_idx = '1'
            else: # その他の時間
                cls_idx = '0'
        # フレームごとにクラスを追記して保存
        f.write(frames_file[i]+','+cls_idx+'\n')
