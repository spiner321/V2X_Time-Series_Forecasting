import shutil
import zipfile
import os
from glob import glob
import pandas as pd

# unzip the file from zip folder to 
zip_path = './data/update/zip'
unzip_on = "./data/update/uncombine"
combine_csv_on = "./data/update/combine"

zip_files = sorted(glob(os.path.join(zip_path, "*.zip")))
for zip_file in zip_files:
    if os.path.isdir(zip_file):
        continue
    zipfile.ZipFile(zip_file).extractall(unzip_on)
    
    # combine the csv files
    unziped_files = sorted(glob(os.path.join(unzip_on, "*.csv")))
    combining_files = pd.concat([pd.read_csv(f) for f in unziped_files])
    combining_files.to_csv(os.path.join(combine_csv_on, os.path.basename(zip_file).replace(".zip", ".csv")), index=False)