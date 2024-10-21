import json
import time
from typing import Union
from numpy import ndarray
from sklearn import metrics as m


def classification_eval_metrics(y_true, y_pred, y_pred_proba=None):
    try:
        roc_auc_ovr_macro = m.roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        roc_auc_ovr_weighted = m.roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        roc_auc_ovo_macro = m.roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
        roc_auc_ovo_weighted = m.roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='weighted')
    except:
        roc_auc_ovr_macro = None
        roc_auc_ovr_weighted = None
        roc_auc_ovo_macro = None
        roc_auc_ovo_weighted = None

    return {
        'Accuracy': m.accuracy_score(y_true, y_pred),
        'Balanced Accuracy': m.balanced_accuracy_score(y_true, y_pred),
        'Precision(Macro)': m.precision_score(y_true, y_pred, average='macro'),
        'Recall(Macro)': m.recall_score(y_true, y_pred, average='macro'),
        'F1(Macro)': m.f1_score(y_true, y_pred, average='macro'),
        'JaccardScore(Macro)': m.jaccard_score(y_true, y_pred, average='macro'), 
        'Precision(Micro)': m.precision_score(y_true, y_pred, average='micro'),
        'Recall(Micro)': m.recall_score(y_true, y_pred, average='micro'),
        'F1(Micro)': m.f1_score(y_true, y_pred, average='micro'),
        'JaccardScore(Micro)': m.jaccard_score(y_true, y_pred, average='micro'),
        'Precision(Weighted)': m.precision_score(y_true, y_pred, average='weighted'),
        'Recall(Weighted)': m.recall_score(y_true, y_pred, average='weighted'),
        'F1(Weighted)': m.f1_score(y_true, y_pred, average='weighted'),
        'JaccardScore(Weighted)': m.jaccard_score(y_true, y_pred, average='weighted'),
        'ROC_AUC_ovr(Macro)': roc_auc_ovr_macro,
        'ROC_AUC_ovr(Weighted)': roc_auc_ovr_weighted,
        'ROC_AUC_ovo(Macro)': roc_auc_ovo_macro,
        'ROC_AUC_ovo(Weighted)': roc_auc_ovo_weighted,
    }


def eval_results(task_name:str, 
                 y_true:Union[ndarray, list], y_pred:Union[ndarray, list], 
                 y_pred_proba:Union[ndarray, list]=None,
                 argparser_params:dict=None, additional_params:dict=None, 
                 common_metrics:bool=True, additional_metrics:dict=None,
                 train_samples:int=None, val_samples:int=None, test_samples:int=None,
                 train_time:float=None, val_time:float=None, test_time:float=None,
                 preprocess_time:float=None, model_load_time:float=None,
                 out:str=None):

    params_dict = dict()
    if argparser_params is not None:
        params_dict.update(argparser_params)
    if additional_params is not None:
        params_dict.update(additional_params)

    metrics_dict = dict()
    if common_metrics:
        metrics_dict.update(classification_eval_metrics(y_true, y_pred, y_pred_proba))
    if additional_metrics is not None:
        metrics_dict.update(additional_metrics)
    

    results_dict = {
        'Task': task_name,
        'WorkTime': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'Params': params_dict,
        'Samples': {
            'Train': train_samples,
            'Val': val_samples,
            'Test': test_samples
        },
        'Time': {
            'Preprocess': preprocess_time,
            'Model load': model_load_time,
            'Train': train_time,
            'Val': val_time,
            'Test': test_time
        },
        'Metrics': metrics_dict
    }

    if out is not None:
        try:
            with open(out, 'w') as f:
                json.dump(results_dict, f, indent=2)
        except Exception as e:
            print(f'Error when writing results to file；{e}')

    return results_dict


