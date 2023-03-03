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
import numpy as np

config_default = AttrDict(
    batch_tfms = TSStandardize(by_sample=True),
    arch_config = {},
    architecture = [LSTM, LSTMPlus, LSTM_FCN, LSTM_FCNPlus, MLSTM_FCN, MLSTM_FCNPlus],
    lr = 1e-3,
    n_epochs = 20,
)

class RunTSAI():
    @staticmethod
    def multivariate_classification(X, y, splits, config=config_default, metrics=accuracy, lr_find=False, load_ckpt = False
                                    , save: str = 'multivariate_classification'):
        "model_input: LSTM, LSTMPlus, MLSTM_Plus, LSTM_FCN"
        
        tfms = [None, [TSCategorize()]]
        check_data(X, y, splits=splits)
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=256, batch_tfms=config["batch_tfms"])
        
        m = create_model(config["architecture"], dls = dls)
        learn = Learner(dls, m, metrics=metrics)
        if lr_find:
            learn.lr_find()
        learn.fit(config["n_epochs"], config["lr"], cbs=SaveModel(monitor='accuracy', fname=save))
        learn.recorder.plot_metrics()
        learn.save_all('classification')
        return learn
    
    @staticmethod
    def multiclass_classification(X, y, split, config, save_path:str = None):
        y=y.tolist()

        tfms = [None, [Categorize]]
        dsets = TSDatasets(X, y, tfms=tfms, splits=split, inplace=True)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], 
                                    batch_tfms=config["batch_tfms"], num_workers=0)
        model = create_model(config["architecture"], dls=dls)
        learn = Learner(dls, model, metrics=[accuracy, F1Score(average='macro'), Precision(average='macro'), Recall(average='macro')])

        # Train model
        learn.fit_one_cycle(config["n_epochs"])
        learn.recorder.plot_metrics()
        
        # is save
        if save_path:
            learn.save_all(save_path)
        return learn
    
    @staticmethod
    def target_label_counter(y):
        result = []
        for i in range(len(y[0])):
            result.append(Counter(y[:,i]))
            print(result[-1])
        return result
    
    @staticmethod
    def plot_confusion_matrix(learn, title:str = None):
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix(title=f'{title} (Unnormalized)')
        interp.plot_confusion_matrix(normalize=True, title=f'{title} (Normalized)')

    @staticmethod
    def multivariate_classification_wandb(X, y, splits, metrics=accuracy, config = config_default, proj_name = "50-1 V2X TSClassification", lr_find=False, load_ckpt = False
                                    , save: str = 'multivariate_classification'):
        # "model_input: LSTM, LSTMPlus, MLSTM_Plus, LSTM_FCN"
        
        with wandb.init(project="multivariate_classification", config=config, name = proj_name) as run:
            # wandb records every epoch
            tfms = [None, [TSCategorize()]]
            check_data(X, y, splits=splits)
            dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
            cbs = [ShowGraph(), WandbCallback(log_preds=False, log_model=False)]
            dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=256, batch_tfms=config["batch_tfms"])
            
            m = create_model(config["architecture"], dls = dls)
            learn = Learner(dls, m, metrics=metrics)
            if lr_find:
                learn.lr_find()
            learn.fit(config["n_epochs"], config["lr"], cbs=cbs)
            learn.recorder.plot_metrics()
            learn.save('classification')
            return learn
        
    @staticmethod
    def minirocket_classification(X, y, splits, epochs=20, model=MiniRocketPlus, lr = 1e-3, metrics=accuracy, lr_find=False, save: str = 'minirocket_classification'):
        tfms = [None, [Categorize()]]
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
        batch_tfms = TSStandardize(by_sample=True)
        dls = get_ts_dls(X, y, bs=256, tfms=tfms, batch_tfms=batch_tfms, splits=splits)
        
        m = create_model(model, dls = dls)
        learn = ts_learner(dls, m, kernel_size=len(X[0]), metrics=metrics)
        if lr_find:
            learn.lr_find()
        learn.fit(epochs, lr)
        learn.recorder.plot_metrics()
        if save:
            learn.save(save)
        return learn

    @staticmethod
    def multivariate_forecasting(X, y, splits, epochs=20, model=LSTM, lr = 1e-3, metrics=mae, lr_find=False):
        "model_input: LSTM, LSTMPlus, MLSTM_Plus, LSTM_FCN"
        splits = TSSplitter()(y)
        tfms = [None, [TSRegression()]]
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
        batch_tfms = TSStandardize(by_var=True)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms, num_workers=0)
        
        m = create_model(model, dls = dls)
        learn = TSForecaster(X, y, splits=splits, batch_tfms=batch_tfms, arch=None, arch_config=dict(fc_dropout=.2), metrics=metrics, bs=512,
                            partial_n=.1, train_metrics=True)
        if lr_find:
            learn.lr_find()
        learn.fit(epochs, lr)
        learn.recorder.plot_metrics()
        learn.save('forecasting')
        return learn
    
