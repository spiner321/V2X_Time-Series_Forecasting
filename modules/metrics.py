from tsai.all import *
import torch


def metrics_multi_common(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh
    
    correct = pred==targ.bool()
    TP = torch.logical_and(correct, (targ==1).bool()).sum()
    TN = torch.logical_and(correct, (targ==0).bool()).sum()
    
    incorrect = pred!=targ.bool()
    FN = torch.logical_and(incorrect, (targ==1).bool()).sum()
    FP = torch.logical_and(incorrect, (targ==0).bool()).sum()
    
    N =  targ.size()[0]
    return N, TP, TN, FP, FN

def precision_multi(inp, targ, **kwargs):
    "Compute precision when `inp` and `targ` are the same size."
    
    N, TP, TN, FP, FN = metrics_multi_common(inp, targ, **kwargs)
    precision = TP/(TP+FP)
    return precision

def recall_multi(inp, targ, **kwargs):
    "Compute recall when `inp` and `targ` are the same size."
    
    N, TP, TN, FP, FN = metrics_multi_common(inp, targ, **kwargs)
    recall = TP/(TP+FN)
    return recall

def specificity_multi(inp, targ, **kwargs):
    "Compute specificity when `inp` and `targ` are the same size."
    
    N, TP, TN, FP, FN = metrics_multi_common(inp, targ, **kwargs)
    specificity = TN/(TN+FP)
    return specificity

def bal_acc_multi(inp, targ, **kwargs):
    "Compute balanced accuracy when `inp` and `targ` are the same size."
    
    N, TP, TN, FP, FN = metrics_multi_common(inp, targ, **kwargs)
    specificity = TN/(TN+FP)
    
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    balanced_accuracy = (TPR+TNR)/2
    return balanced_accuracy

def Fbeta_multi(inp, targ, beta=1.0, **kwargs):
    "Compute Fbeta when `inp` and `targ` are the same size."
    
    N, TP, TN, FP, FN = metrics_multi_common(inp, targ, **kwargs)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    beta2 = beta*beta
    
    if precision+recall > 0:
        Fbeta = (1+beta2)*precision*recall/(beta2*precision+recall)
    else:
        Fbeta = 0
    return Fbeta

def F1_multi(*args, **kwargs):
    return Fbeta_multi(*args, **kwargs)  # beta defaults to 1.0