import os
import zipfile
import pickle
import pathlib as pl
import argparse
from typing import List, Tuple, Dict, Union, Optional
import numpy as np

class V2XPreprocessing:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir

    @staticmethod
    def unzip_files(folder):
        # 해당 폴더에 있는 모든 zip 파일을 각 폴더 명으로 압축 해제
        zip_list = [file for file in os.listdir(folder) if file.endswith(".zip")]
        for zip_file in zip_list:
            zip_path = os.path.join(folder, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(folder, zip_file[:-4]))
    
    @staticmethod
    def get_file_list(data_dir) -> Tuple[List[str], List[str]]:
        # if data_dir is a folder, return all csv and json files in the folder
        csv_files, json_files = [], []
        for file in pl.data_dir.rglob("*"):
            if file.suffix == ".csv":
                csv_files.append(str(file))
            elif file.suffix == ".json":
                json_files.append(str(file))
        return csv_files, json_files
    
    @staticmethod
    def get_csv_data(csv_folder: str) -> List[str]:
        csv_files = sorted([str(p) for p in pl.Path(csv_folder).rglob("*.csv")])
        return csv_files

    @staticmethod
    def get_json_data(json_folder: str) -> List[str]:
        json_files = sorted([str(p) for p in pl.Path(json_folder).rglob("*.json")])
        return json_files
    
    @staticmethod
    def make_npy_csv(data_list: List[str], output_dir: str):
        pass
        X_sum = np.empty((0, 50, 1))
        
        

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", '-d', type=str, help="data folder path", default=None)
    arg.add_argument("--output_dir", '-o', type=str, help="output folder path", default=None)
    arg.add_argument("--unzip", '-u', type=str, help="압축 파일이 있는 경로 입력. 각 모든 압축 파일을 각 압축파일명으로 압축해제함.", default=None)
    args = arg.parse_args()

    if args.unzip:
        V2XPreprocessing.unzip_files(arg.unzip)
        
    if args.data_dir:
        csv_list, json_list = list(V2XPreprocessing.get_file_list(args.data_dir))
        print(f"csv file count: {len(csv_list)}, json file count: {len(json_list)}")

    