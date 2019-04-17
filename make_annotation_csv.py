import pandas as pd
import numpy as np
from glob import glob
import os
import shutil

# パスを指定
csv_dir_path = 'raw_data/digtime_over3sec'
frame_dir_path = 'raw_data/frames'
# 保存ファイル名
savename = 'annotation_all_over3sec'
# データ数のtrainに使う割合
split_rate = 0.9


# 保存ファイル生成
with open(savename+'train.csv', 'w') as f:
    f.write('frame,class\n')
with open(savename+'test.csv', 'w') as f:
    f.write('frame,class\n')

# 全サンプルのdigtime.csv読み込み
label_csvs = glob(csv_dir_path+'/*')
np.random.shuffle(label_csvs)
# train/testに分ける
label_csvs_train = label_csvs[:int(len(label_csvs)*split_rate)]
label_csvs_test = label_csvs[int(len(label_csvs)*split_rate):]

def time_to_sec(time):
    '''時刻を秒数へ変換'''
    if ':' in time: # excel表記
        min, sec = time.split(':')
    elif '-' in time: # ファイル名表記
        min, sec = time.split('-')
    min, sec = float(min), float(sec)
    sec = min*60 + sec
    return sec

def annotation_from_csv(label_csv, savename): # trainとtestに同じ作業をするので関数化
    '''フレームにラベリングする'''
    # 対象フレーム区間(ラベル"1")
    df = pd.read_csv(label_csv)
    start_time = df['start'].to_list()
    end_time = df['end'].to_list()

    # 入力データ
    label_csv_basename = os.path.basename(label_csv).split('.')[0]
    frames_file = glob(os.path.join(frame_dir_path, label_csv_basename, 'frames/*'))
    frames_file = sorted(frames_file) # 時系列順にソート

    # 秒へ変換
    start_time = [time_to_sec(t) for t in start_time] # ラベル"1"
    end_time = [time_to_sec(t) for t in end_time]
    frames_time = [os.path.basename(f) for f in frames_file] # ファイル名取得
    frames_time = [os.path.splitext(f)[0] for f in frames_time] # 拡張子を取り除く
    frames_time = [time_to_sec(f) for f in frames_time] # 入力データ

    # 穴掘り時間のフレームをラベリング
    with open(savename, 'a') as f:
        for i, ft in enumerate(frames_time): # まずフレームを固定して
            cls_idx = '0'
            for st, et in zip(start_time, end_time): # 穴掘り時間に該当するか探してラベリング
                if st <= ft and et >= ft: # 穴掘りしている時間
                    cls_idx = '1'
                    f.write(frames_file[i]+','+cls_idx+'\n')
                    break # 穴掘りを見つけたら次のフレームの検索へ
            if cls_idx == '0': # 該当しなかったら
                f.write(frames_file[i]+','+cls_idx+'\n')

# train
for label_csv in label_csvs_train:
    annotation_from_csv(label_csv, savename+'train.csv')
# test
for label_csv in label_csvs_test:
    annotation_from_csv(label_csv, savename+'train.csv')
