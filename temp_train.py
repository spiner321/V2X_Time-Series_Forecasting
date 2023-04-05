import numpy as np
import cv2
import torch
import pandas as pd
import re
import pickle as pkl
import json
import glob
import shutil
import os
import json
import pathlib as pl
from typing import List, Tuple, Dict, Union, Optional
from tqdm import trange, tqdm
from glob import glob
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from modules.preprocessing import V2XPreprocessing
from modules.v2x import V2XData,V2XDataLabeled
from modules.metrics import *
from tsai.all import *
from modules.RunTSAI import RunTSAI
from sklearn.utils import shuffle

# # json file 전처리 (미사용)
# def preprocessing_json(json_file):
#     jf = pd.read_json(json_file)
#     jf = jf.T.drop('Path', axis=0).reset_index().drop(['index', 'Vehicle_ID'], axis=1)
    
#     return jf



if __name__=='__main__':
    
    # 데이터 준비

    src = '/data/NIA50/docker/50-1/data/8to11/scaled'
    dst_h = 'pth/8to11/hazard/'
    dst_s = 'pth/8to11/speed/'
    dst_t = 'pth/8to11/turn/'

    X_8to11 = np.load(f'{src}/X_8to11.npy')
    y_8to11 = np.load(f'{src}/y_8to11.npy', allow_pickle=True)

    y_8to11_hazard = y_8to11[:, 0]
    y_8to11_speed = y_8to11[:, 2]
    y_8to11_turn = y_8to11[:, 3]

    # 데이터 분할
    split_h = get_splits(y_8to11_hazard, shuffle=True, valid_size=0.1, test_size=0.1, random_state=44)
    split_s = get_splits(y_8to11_speed, shuffle=True, valid_size=0.1, test_size=0.1, random_state=44)
    split_t = get_splits(y_8to11_turn, shuffle=True, valid_size=0.1, test_size=0.1, random_state=44)

    # 모델 설정
    config = AttrDict(
        batch_tfms = TSStandardize(),
        architecture = LSTM, # LSTM, LSTM_FCNPlus, MLSTM_Plus
        n_epochs = 10,)

    RunTSAI.target_label_counter(y_8to11)

    # 훈련
    learn8to10_hazard = RunTSAI.multiclass_classification(X_8to11, 
                                                        y_8to11_hazard, 
                                                        split=split_h,
                                                        config=config, 
                                                        save_path=dst_h)

    # 훈련
    learn8to10_speed = RunTSAI.multiclass_classification(X_8to11, 
                                                        y_8to11_speed, 
                                                        splits=split_h, 
                                                        config=config, 
                                                        save_path=dst_s)                               

    # 훈련
    learn8to10_turn = RunTSAI.multiclass_classification(X_8to11, 
                                                        y_8to11_turn, 
                                                        splits=split_h, 
                                                        config=config, 
                                                        save_path=dst_t)




     # 데이터 준비

    src = '/data/NIA50/docker/50-1/data/8s/scaled'
    dst_h = 'pth/8s/hazard/'
    dst_s = 'pth/8s/speed/'
    dst_t = 'pth/8s/turn/'

    X_8s = np.load(f'{src}/X_8s.npy')
    y_8s = np.load(f'{src}/y_8s.npy', allow_pickle=True)

    y_8s_hazard = y_8s[:, 0]
    y_8s_speed = y_8s[:, 2]
    y_8s_turn = y_8s[:, 3]

    # 데이터 분할
    split_h = get_splits(y_8s_hazard, shuffle=True, valid_size=0.1, test_size=0.1, random_state=44)
    split_s = get_splits(y_8s_speed, shuffle=True, valid_size=0.1, test_size=0.1, random_state=44)
    split_t = get_splits(y_8s_turn, shuffle=True, valid_size=0.1, test_size=0.1, random_state=44)

    # 모델 설정
    config = AttrDict(
        batch_tfms = TSStandardize(),
        architecture = LSTM, # LSTM, LSTM_FCNPlus, MLSTM_Plus
        n_epochs = 10,)

    RunTSAI.target_label_counter(y_8s)

    # 훈련
    learn8s_hazard = RunTSAI.multiclass_classification(X_8s, 
                                                    y_8s_hazard, 
                                                    split=split_h,
                                                    config=config, 
                                                    save_path=dst_h)

    # 훈련
    learn8s_speed = RunTSAI.multiclass_classification(X_8s, 
                                                    y_8s_speed, 
                                                    splits=split_s, 
                                                    config=config, 
                                                    save_path=dst_s)                      

    # 훈련
    learn8s_turn = RunTSAI.multiclass_classification(X_8s, 
                                                    y_8s_turn, 
                                                    splits=split_t, 
                                                    config=config, 
                                                    save_path=dst_t)                      