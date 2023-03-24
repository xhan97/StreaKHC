import os
import pandas as pd
import numpy as np
import random


# metric
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def ad_metric(tree_ad_score, pos_label=1):
    res = pd.DataFrame.from_records(tree_ad_score, columns =['pid', 'y_true', 'y_score'])
    aucroc = roc_auc_score(y_true=res["y_true"], y_score=res['y_score'])*100
    aucpr = average_precision_score(y_true=res["y_true"], y_score=res['y_score'], pos_label=pos_label)
    #f1 = f1_score(y_true=y_true, y_score=y_score,)

    return {'aucroc':aucroc}