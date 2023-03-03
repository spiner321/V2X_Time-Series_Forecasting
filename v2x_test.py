from tsai.all import *
import sklearn.metrics as skm
import numpy as np
import pandas as pd
import sys
import argparse
from matplotlib import pyplot as plt
from collections import Counter
from datetime import datetime

from modules.utils import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--class_name", type=str, default="False", help="turn, lane, speed, hazard")
parser.add_argument("--continue_test", type=str, default="False", help="path to the previous log file to continue inference from the last index")
parser.add_argument("--average", type=str, default="macro", help="macro, micro, weighted")
args = parser.parse_args()

if args.continue_test != "False":
    args.class_name = args.continue_test.split(".")[0].split("_")[-3]

LOG_START_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
DATANUM = 900000
AVERAGE = "macro"
LOG_START_INX = 0

# load data 20220801 ~ 20220831
X, y = np.load("data/X_sum_all.npy"), np.load("data/y_sum_all.npy")
y_turn, y_lane, y_speed, y_hazard = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

ct_turn, ct_lane, ct_speed, ct_hazard = Counter(y_turn), Counter(y_lane), Counter(y_speed), Counter(y_hazard)

def start_log(classes: dict):
    write_txt_log("=====================================")
    write_txt_log(f"Test data length: 180000")
    write_txt_log(f"Class types: {classes}")
    write_txt_log("=====================================")
    write_txt_log("V2X Time Series Classification Result")
    write_txt_log(f"Average: {AVERAGE}")
    write_txt_log(f"idx: Target, Prediction, Precision, Recall, F1_Score")

def write_txt_log(*args_method):
    with open(f"output/result_{args.class_name}_{LOG_START_TIME}.txt", "a") as f:
        current_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        for arg in args_method:
            print(f"[{current_time}] {arg}")
            f.writelines(f"[{current_time}] {arg}\n")

# 0: False, 1: Right, 2: Reverse, 3: Left -> 0: False, 1: Right, 2: Left, 3: Reverse
def permute_turn(test_targets, test_preds):
    test_targets = test_targets.tolist()
    test_preds = test_preds.tolist()

    for idx in range(len(test_targets)):
        if test_targets[idx] == 3:
            test_targets[idx] = 2
        elif test_targets[idx] == 2:
            test_targets[idx] = 3

    for idx in range(len(test_preds)):
        if test_preds[idx] == 3:
            test_preds[idx] = 2
        elif test_preds[idx] == 2:
            test_preds[idx] = 3
    return test_targets, test_preds

# 0: Acc, 1: False, 2: Hbrk -> 0: False, 1: Acc, 2: Hbrk
def permute_speed(test_targets, test_preds):
    test_targets = test_targets.tolist()
    test_preds = test_preds.tolist()

    for idx in range(len(test_targets)):
        if test_targets[idx] == 0:
            test_targets[idx] = 1
        elif test_targets[idx] == 1:
            test_targets[idx] = 0

    for idx in range(len(test_preds)):
        if test_preds[idx] == 0:
            test_preds[idx] = 1
        elif test_preds[idx] == 1:
            test_preds[idx] = 0
    return test_targets, test_preds

def get_preds_csv(learner, classes):
    dls = learner.dls
    valid_dl = dls.valid

    # inference
    _, test_targets, test_preds = learner.get_preds(dl=valid_dl, with_decoded=True, save_preds=None, save_targs=None)
    if args.class_name == "turn":
        test_targets, test_preds = permute_turn(test_targets, test_preds)
    elif args.class_name == "speed":
        test_targets, test_preds = permute_speed(test_targets, test_preds)

    df = pd.DataFrame(columns=["Target", "Prediction", "Recall", "Precision", "F1_Score"])
    df_rows = []
    
    if not args.continue_test:
        write_txt_log("=====================================")
        write_txt_log("Time Series Classification Report")
        with open(f"output/result_{args.class_name}_{LOG_START_TIME}.txt", "a") as f:
            sys.stdout = f
            print(skm.classification_report(test_targets, test_preds, target_names=list(classes.values()), zero_division=0, digits=4))
            sys.stdout = sys.__stdout__
        
            start_log(classes)
        
    for idx in range(LOG_START_INX, len(test_targets)):
        target_class, pred_class = classes[int(test_targets[idx])], classes[int(test_preds[idx])]
        test_precision = round(skm.precision_score(test_targets[:idx+1], test_preds[:idx+1], average = f"{AVERAGE}"), 4)
        test_recall = round(skm.recall_score(test_targets[:idx+1], test_preds[:idx+1], average = f"{AVERAGE}"), 4)
        test_f1 = round(skm.f1_score(test_targets[:idx+1], test_preds[:idx+1], average = f"{AVERAGE}"), 4)
        
        write_txt_log(f"[{idx+1}] target: {target_class}, pred: {pred_class}, precision: {test_precision}, recall: {test_recall}, f1 score: {test_f1}")
        
        df_rows.append([target_class, pred_class, test_precision, test_recall, test_f1])
        
    df = df.append(df_rows, ignore_index=True)

    write_txt_log("Confusion Matrix")
    with open(f"output/result_{args.class_name}_{LOG_START_TIME}.txt", "a") as f:
        sys.stdout = f
        print(skm.confusion_matrix(test_targets, test_preds))
        sys.stdout = sys.__stdout__
    
    plot_confusion_matrix(test_targets, test_preds, classes=list(classes.values()), normalize=True, title='Normalized confusion matrix').savefig(f"output/confusion_matrix_{args.class_name}_{LOG_START_TIME}_{int(True)}.png")
    plot_confusion_matrix(test_targets, test_preds, classes=list(classes.values()), normalize=False, title='Confusion matrix').savefig(f"output/confusion_matrix_{args.class_name}_{LOG_START_TIME}_{int(False)}.png")
    df.to_excel(f"output/result_{args.class_name}_{LOG_START_TIME}.xlsx", index=False)
    df.to_csv(f"output/result_{args.class_name}_{LOG_START_TIME}.csv", index=False)
    
    write_txt_log("")
    write_txt_log("")
            

def load_model_v2x(arg):
    models_folder = {"turn": "turn_20221226_0955", 
                     "speed": "speed_20221226_1202", 
                     "hazard": "hazard_20221226_1809"} # 모델 저장 폴더
    
    if arg == "turn":
        classes = {0: "False", 1: "Right", 2: "Left", 3: "Reverse"}
        if args.continue_test == "False":
            write_txt_log(f"{arg.title()} counter: {' '.join(f'{k}: {v}' for k, v in ct_turn.items())}")
            write_txt_log(f"{arg.title()} percentage: {' '.join(f'{k}: {v / DATANUM:.4f}' for k, v in ct_turn.items())}")
    elif arg == "speed":
        classes = {0: "False", 1: "Acc", 2: "Hbrk"}
        if args.continue_test == "False":
            write_txt_log(f"{arg.title()} counter: {' '.join(f'{k}: {v}' for k, v in ct_speed.items())}")
            write_txt_log(f"{arg.title()} percentage: {' '.join(f'{k}: {v / DATANUM:.4f}' for k, v in ct_speed.items())}")
    elif arg == "hazard":
        classes = {0: "False", 1: "True"}
        if args.continue_test == "False":
            write_txt_log(f"{arg.title()} counter: {' '.join(f'{k}: {v}' for k, v in ct_hazard.items())}")
            write_txt_log(f"{arg.title()} percentage: {' '.join(f'{k}: {v / DATANUM:.4f}' for k, v in ct_hazard.items())}")
    else :
        write_txt_log("Wrong class name")
        assert print("Wrong class name")
    
    learner = load_learner_all(path=f"models/{models_folder[arg]}", dls_fname=f'dls_{arg}', model_fname=f'model_{arg}_MLSTM_FCNPlus', learner_fname=f'learner_{arg}')
    return learner, classes

if __name__ == "__main__":
    
    # 로그 중간 시작 시
    if args.continue_test != "False":
        LOG_START_TIME = "_".join(args.continue_test.split(".")[0].split("_")[-2:])
        
        with open(args.continue_test, "r") as f:
            lines = f.readlines()
            idx_start = lines[-1].find('[', 1)
            idx_end = lines[-1].find(']', idx_start)
            LOG_START_INX = int(lines[-1][idx_start+1:idx_end])
        
        # 로그 파일 이름이 잘못된 경우 확인
        if not os.path.exists(args.continue_test):
            write_txt_log("Wrong log file name to continue test.")
            assert print("Wrong log file name to continue test.")
            exit()
    else:
        write_txt_log(f"python v2x_exc.py --class_name {args.class_name}") # 실행 명령어 기록 
    
    learner, classes = load_model_v2x(args.class_name) # 모델 불러오기
    get_preds_csv(learner, classes) # 결과 로그 및 excel 파일 생성