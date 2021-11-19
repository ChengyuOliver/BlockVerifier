# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns', 12)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fundchartv9 import pic


def quick_merge(df1: pd.DataFrame, df2: pd.DataFrame, 
                on: list, feat: list, fillna=-1):
    assert type(on) == list and type(feat) == list
    if len(on) == 1:
        d = {}
        for item in df2[on+feat].values:
            if item[0] not in d:
                d[item[0]] = item[1:]
        df1['temp'] = df1[on].values.tolist()
        for ix, col in enumerate(feat):
            df1[col] = df1['temp'].apply(
                    lambda x:d[x[0]][ix] if x[0] in d else fillna)
        del df1['temp']
    elif len(on) == 2:
        d = {}
        for item in df2[on+feat].values:
            if item[0] not in d:
                d[item[0]] = {}
            if item[1] not in d[item[0]]:
                d[item[0]][item[1]] = item[2:]
        df1['temp'] = df1[on].values.tolist()
        for ix, col in enumerate(feat):
            df1[col] = df1['temp'].apply(
                    lambda x:d[x[0]][x[1]][ix] if x[0] in d and x[1] in d[x[0]] else fillna)
        del df1['temp']
    elif len(on) == 3:
        d = {}
        for item in df2[on+feat].values:
            if item[0] not in d:
                d[item[0]] = {}
            if item[1] not in d[item[0]]:
                d[item[0]][item[1]] = {}
            if item[2] not in d[item[0]][item[1]]:
                d[item[0]][item[1]][item[2]] = item[3:]
        df1['temp'] = df1[on].values.tolist()
        for ix, col in enumerate(feat):
            df1[col] = df1['temp'].apply(
                    lambda x:d[x[0]][x[1]][x[2]][ix] if 
                    x[0] in d and 
                    x[1] in d[x[0]] and 
                    x[2] in d[x[0]][x[1]] 
                    else fillna)
        del df1['temp']
    else:
        pass
    return df1



func_lst = [np.max, np.min, np.mean, np.sum, np.std]
def get_feat(df):
    feat = []
# =============================================================================
#     feat.append(df.shape[0])
# =============================================================================
    # df['Investment_add_Payment'] = df['Investment transnum'] + df['Payment transnum']
    for f in func_lst:
        for col in ['Total investment', 'Total payment', 'Investment transnum', 'Payment transnum']:
            feat.append(f(df[col]))
    return feat

def get_feat_fine_grained(cat, fn, num=None):
    n = 15
    new_fn = fn[:-7]
    # lst = os.listdir(f'./input/transaction v6/{cat}')
    lst = os.listdir(f'./input/1.7/{cat}')
    for item in lst:
        item = item.split('.')[0]
        if item == new_fn:
            break
    if len(lst)==0 or item != new_fn:
        return [0]*n
    
    df1 = pd.read_csv(f'./input/1.7/{cat}/{fn}')
    # df2 = pd.read_csv(f'./input/transaction v6/{cat}/{new_fn}.csv').rename(
    #         columns={'From': 'Personname'})    
    df2 = pd.read_csv(f'./input/1.7/{cat}/{new_fn}.csv').rename(
            columns={'From': 'Personname'})
    df2 = df2.sort_values(by='Block').reset_index(drop=True)
    if num is None:
        num = len(df1)
    df1 = df1.loc[:num]
    people_set = set(df1.Personname)
    df2 = df2.loc[df2.Personname.apply(lambda x: 1 if x in people_set else 0)==1].reset_index(drop=True)
    last_people = df1.Personname[num-1]
    max_block = df2.Block.loc[df2.Personname==last_people].max()
    df2 = df2.loc[df2.Block<max_block].reset_index(drop=True)
    
    Personname_set = set(df1.Personname)
    df2 = df2.loc[df2.Personname.apply(lambda x: 1 if x in Personname_set else 0)==1].reset_index(drop=True)
    df2.Value = df2.Value.fillna(0)
    feature = []
    
    tmp = df2.groupby('Personname').size().reset_index()
    tmp.columns = ['Personname', 'cnt']
    df1 = quick_merge(df1, tmp, on=['Personname'], feat=['cnt'])
    for f in func_lst:
        feature.append(f(df1['cnt']))
    
    try:
        tmp = df2.groupby('Personname')['Value'].agg(np.std).reset_index()
    except:
        return [0]*n
    tmp.columns = ['Personname', 'Value_std']
    df1 = quick_merge(df1, tmp, on=['Personname'], feat=['Value_std'])
    for f in func_lst:
        feature.append(f(df1['Value_std']))
    
    df2.Block = df2.Block - df2.Block.min()
    try:
        tmp = df2.groupby('Personname')['Block'].agg(np.ptp).reset_index()
    except:
        return [0]*n
    tmp.columns = ['Personname', 'Block_ptp']
    tmp.Block_ptp /= df2.Block.max()
    df1 = quick_merge(df1, tmp, on=['Personname'], feat=['Block_ptp'])
    for f in func_lst:
        feature.append(f(df1['Block_ptp']))
    
# =============================================================================
#     df2.Block = df2.Block - df2.Block.min()
#     try:
#         tmp = df2.groupby('Personname')['Flag'].agg(np.sum).reset_index()
#     except:
#         return [0]*n
#     tmp.columns = ['Personname', 'Flag_sum']
#     df1 = quick_merge(df1, tmp, on=['Personname'], feat=['Flag_sum'])
#     for f in func_lst:
#         feature.append(f(df1['Flag_sum']))
# =============================================================================
    
    
    return feature

from tqdm import tqdm
cat_lst = ['Identity', 'Exchanges', 'Security', 'Energy', 'Development', 'Property',
           'Health', 'Gambling', 'High_risk', 'Storage', 'Social', 'Wallet', 'Governance', 'Finance',
           'Games', 'Insurance', 'Marketplaces', 'Media']
train = []
for cat in tqdm(cat_lst):
    for fn in os.listdir(f'./input/1.7/{cat}'):
        if not fn.endswith('.csv'):
            continue
        df = pd.read_csv(f'./input/1.7/{cat}/{fn}')
        train.append([f'{cat}_{fn}'] + get_feat(df) + get_feat_fine_grained(cat, fn))
feature = [f'feat_{i}' for i in range(len(train[0])-1)]
feature = [f'{col}_{f}' for f in ['max', 'min', 'mean', 'sum', 'std'] 
for col in ['Total investment', 'Total payment', 'Investment transnum', 'Payment transnum', 
            'cnt', 'Value_std', 'Block_ptp']] # , 'Flag_sum'
train = pd.DataFrame(train, columns=['fn']+feature)
train['cat'] = train['fn'].apply(lambda x: x.split('_')[0])
train['cat'] = np.where(train['cat']=='High', 'High_risk', train['cat'])

train['cat2'] = train['cat'].apply(lambda x: 1 if x=='High_risk' else 0)
print(train['cat2'].mean())
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder().fit(train['cat'])
train['cat'] = enc.transform(train['cat'])

# =============================================================================
# 
# =============================================================================
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


train2 = train.copy()

oof = np.zeros(train.shape[0])
feat_importance = np.zeros(len(feature))
n_rounds = 100
tk = tqdm(range(n_rounds))
for seed in tk:
    skf = StratifiedKFold(n_splits=8, random_state=3+seed, shuffle=True)
    for fold, (tr_ix, val_ix) in enumerate(skf.split(train, train['cat2']), 1):
        tr, val = train[feature].loc[tr_ix].values, train[feature].loc[val_ix].values
        y_tr, y_val = train['cat2'].loc[tr_ix].values, train['cat2'].loc[val_ix].values
        clf = lgb.LGBMClassifier(n_estimators=10000, n_jobs=-1, num_leaves=31,
                                 reg_alpha=0.2, reg_lambda=0.2, colsample_bytree=0.7) # 避免烧笔记本
        clf.fit(tr, y_tr, eval_set=[(val, y_val)], verbose=False, early_stopping_rounds=50)
        oof[val_ix] += clf.predict_proba(val)[:, 1] / n_rounds
        feat_importance += clf.feature_importances_
    tk.set_postfix(auc=roc_auc_score(train['cat2'], oof))
print('train auc = {:.4f}'.format(roc_auc_score(train['cat2'], oof)))

def get_max_f1(y, oof):
    from sklearn.metrics import f1_score
    import warnings
    warnings.filterwarnings('ignore')
    MAX = 0
    T = 0
    for i in range(100):
        tp = f1_score(y, np.where(oof>i/100, 1, 0))
        if tp > MAX:
            T = i
        MAX = max([MAX, tp])
    return MAX, T
print(get_max_f1(train['cat2'], oof))
# =============================================================================
# 
# =============================================================================

# =============================================================================
# skf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
# n_class = len(set(train['cat']))
# oof = np.zeros((train.shape[0], n_class))
# for fold, (tr_ix, val_ix) in enumerate(skf.split(train, train['cat']), 1):
#     print(f'fold {fold}')
#     tr, val = train[feature].loc[tr_ix].values, train[feature].loc[val_ix].values
#     y_tr, y_val = train['cat'].loc[tr_ix].values, train['cat'].loc[val_ix].values
#     clf = lgb.LGBMClassifier(n_estimators=10000, n_jobs=-1, num_leaves=31) # 避免烧笔记本
#     clf.fit(tr, y_tr, eval_set=[(val, y_val)], verbose=False, early_stopping_rounds=50)
#     oof[val_ix] = clf.predict_proba(val)
# from sklearn.metrics import accuracy_score
# print('train acc = {:.4f}'.format(accuracy_score(train['cat'], np.argmax(oof, axis=1))))
# =============================================================================



'''
# train acc = 0.3197
# train auc = 0.8156

z = sorted(zip(feature, feat_importance), key=lambda x: x[1], reverse=False)
feat = [z[i][0] for i in range(len(z))]
feat_importance = [z[i][1] for i in range(len(z))]
plt.figure(figsize=(20, 20))
plt.barh(feature, feat_importance)
plt.yticks(size=15)
plt.title('feature importance', size=20)
plt.savefig('./feature_importance.jpg')
plt.show()



from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train.cat2, oof)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC curve')
plt.show()
'''

# import matplotlib.pyplot as plt

# plt.subplot(2,1,1)
# pic(f'./input/dapp_722/High_risk/{tmp}/', tmp)
# plt.subplot(2,2,2)
# plt.plot(prob)
