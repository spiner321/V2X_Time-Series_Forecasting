from glob import glob
from tqdm import tqdm
import os
import pandas as pd
from tsai.all import *
from tsai.models.MINIROCKET import *
from fastai.torch_core import default_device
from fastai.metrics import accuracy
from fastai.callback.tracker import ReduceLROnPlateau
from tsai.data.all import *
from tsai.learner import *
from sklearn.model_selection import train_test_split
from tsai.basics import *
from tsai.data.external import *
from tsai.data.preprocessing import *
from torch.utils.data import DataLoader, Dataset
import numpy as np

DROP_COLS = ['VEHICLE_CLASS', 'VEHICLE_TYPE', 'Turn', 'Change', 'Speed', 'ISSUE_DATE', 'X', 'Y']
TARGET_COLS = ["Hazard"]
INDEX_COL = "ISSUE_DATE"


class V2XData():
    def __init__(self, dataset_path = None, dataset_scenario: list = None, drop_cols: list = None, 
                 target_cols: list = None, test_size: float = 0.5, random_state: int = None,
                 hazard_thr: int = 1):
        self.data_path = dataset_path
        if dataset_scenario:
            self.dataset_scenario = dataset_scenario.copy()
        else:
            self.dataset_scenario = sorted(glob.glob(os.path.join(self.data_path, '*.csv')))
        
        
        if not drop_cols:
            self.drop_cols = DROP_COLS.copy()    
        else:
            self.drop_cols = drop_cols
        
        if not target_cols:
            self.target_cols = TARGET_COLS.copy()
        else:
            self.target_cols = target_cols
        
        self.test_size = test_size
        self.random_state = random_state
        self.files_num = len(self.dataset_scenario)
        
        if hazard_thr < 1:
            assert hazard_thr > 0, "hazard_thr must be greater than 0"
        else: 
            self.hazard_thr = hazard_thr
        
        if self.files_num:
            print(f"loaded {len(self.dataset_scenario)} files")
            print(self.dataset_scenario[:5], "...")
            
        
    def __getitem__(self, index, print_size=True):
        df = pd.read_csv(self.dataset_scenario[index]).drop(labels = self.drop_cols, axis=1)
        df.dropna(0, inplace = True)
        if print_size: print(f'df[{index}] shape: {df.shape}')
        # print(df.info())
        df_filtered = pd.crosstab(df['scene'], df['Hazard'])
        df = df.groupby(df['scene']).filter(lambda x: len(x) == 10)
        df.reset_index(drop=True, inplace=True)
        change_list = list(df_filtered[df_filtered[True] >= self.hazard_thr].index)
        df.loc[df['scene'].isin(change_list), 'Hazard'] = True
        
        y = df['Hazard'].iloc[::10]
        y = y.astype(int)
        y = y.to_numpy()
        
        X = df.groupby(df["scene"]).apply(lambda x: x.drop(["scene", "Hazard"], axis=1).values)
        X = np.array(X.tolist())
        X = np.array([scene.transpose() for scene in X])
        
        splits = self.get_splits(X, test_size = self.test_size, random_state = self.random_state)
        
        return X, y, splits, df
    
    def get_splits(self, X, test_size: float = 0.5, random_state: int = None):
        X_train, X_valid = train_test_split(X, test_size=test_size, random_state=random_state)
        splits = get_predefined_splits(X_train, X_valid)
        return splits
    
    @staticmethod
    def get_data_info(X, y, splits = None):
        print('Dataset Info is...')
        print(f'X shape: {X.shape}, y shape: {y.shape}') 
        if(splits):
            print(f'splits: (train: (#{len(splits[0])})({splits[0][0]}, ...)) (test: (#{len(splits[1])})({splits[1][0]}, ...))')
        print(f'# True in y: {np.unique(y, return_counts=True)}')
        print('Dataset Info is done.')
    
    def get_all_item(self, is_test = False, print_size=True):
        X_sum, y_sum = [], []
        df_sum = pd.DataFrame()
        files_num = len(self.dataset_scenario)
        if is_test == True:
            files_num = int(files_num - 0.2*files_num)
        print(f'files index: 0 ~ {files_num}')
        
        for idx in tqdm(range(files_num)):
            X, y, _, df = self.__getitem__(idx, print_size=False)
            X_sum.append(X)
            y_sum.append(y)
            df_sum = pd.concat([df_sum, df])
            
            
        X_sum = np.concatenate(X_sum)
        y_sum = np.concatenate(y_sum)
        df_sum.reset_index(drop=True, inplace=True)
        splits = self.get_splits(X_sum, test_size = self.test_size, random_state = self.random_state)
        print(f'X_sum shape: {X_sum.shape}, y_sum shape: {y_sum.shape}')
        return X_sum, y_sum, splits, df_sum
    
DROP_COLS_L = ['ISSUE_DATE', 'VEHICLE_ID', 'VEHICLE_TYPE']

# class V2XDataLabeled(V2XData):
#     def __init__(self, dataset_path = None, dataset_scenario: list = None, drop_cols: list = None, 
#                  target_cols: list = None, test_size: float = 0.5, random_state: int = None):
#         super().__init__(dataset_scenario, drop_cols, target_cols, test_size, random_state)

class V2XDataLabeled():
    def __init__(self, dataset_path, condition: str, dataset_scenario: list = None, drop_cols: list = None, 
                 target_cols: list = None, test_size: float = 0.5, random_state: int = None):
        self.condition = condition
        self.scenario_path = os.path.join(dataset_path, condition)
        
        if dataset_scenario:
            self.dataset_scenario = dataset_scenario.copy()
        else:
            self.dataset_scenario = sorted(glob.glob(os.path.join(self.scenario_path, '*')))
        self.dataset_scenario = [file for file in self.dataset_scenario if os.path.isdir(file)]
        
        if not drop_cols:
            self.drop_cols = DROP_COLS_L.copy()
        else:
            self.drop_cols = drop_cols.copy()
            
        if target_cols:
            self.target_cols = target_cols.copy()
        else:
            self.target_cols = None
            
        self.test_size = test_size
        self.random_state = random_state
        
        
        print(f"loaded {len(self.dataset_scenario)} scenarios")
        print(self.dataset_scenario[:5], "...")
            
    def __len__(self):
        return len(self.dataset_scenario)
        
    def __getitem__(self, index, print_size=True):
        csv_files = sorted(glob.glob(os.path.join(self.dataset_scenario[index], '*.csv')))
        json_files = sorted(glob.glob(os.path.join(self.dataset_scenario[index].replace("/C", ""), '*.json')))
        
        df = pd.DataFrame()
        X = []
        for csv_file in csv_files:
            read_df = pd.read_csv(csv_file)
            df = pd.concat([df, read_df])
            X.append(read_df.drop(labels = self.drop_cols, axis=1).values)
        X = np.array(X)
        X = np.array([scene.transpose() for scene in X])
        
        df.fillna(0, inplace = True)
        
        if json_files:
            y = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    y.append([json_data["Annotation"]['Turn'], json_data["Annotation"]['Lane'], json_data["Annotation"]["Speed"], json_data["Annotation"]['Hazard']])
            
            y = np.array(y)
        else:
            y = np.zeros((len(X), 4))
        print(X, self.test_size, self.random_state)
        splits = self.get_splits(X, test_size = self.test_size, random_state = self.random_state)
        
        return X, y, splits, df
    
    def get_splits(self, X, test_size: float = 0.5, random_state: int = None):
        X_train, X_valid = train_test_split(X, test_size=test_size, random_state=random_state)
        splits = get_predefined_splits(X_train, X_valid)
        return splits
    
    def get_all_item(self, is_test = False, print_size=True):
        X_sum, y_sum = [], []
        df_sum = pd.DataFrame()
        files_num = len(self.dataset_scenario)
        if is_test == True:
            files_num = int(files_num - 0.2*files_num)
        print(f'files index: 0 ~ {files_num}')
        
        for idx in tqdm(range(files_num)):
            X, y, _, df = self.__getitem__(idx, print_size)
            X_sum.append(X)
            y_sum.append(y)
            df_sum = pd.concat([df_sum, df])
            
            
        X_sum = np.concatenate(X_sum)
        y_sum = np.concatenate(y_sum)
        df_sum.reset_index(drop=True, inplace=True)
        splits = self.get_splits(X_sum, test_size = self.test_size, random_state = self.random_state)
        print(f'X_sum shape: {X_sum.shape}, y_sum shape: {y_sum.shape}')
        return X_sum, y_sum, splits, df_sum