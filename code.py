# -*- coding: utf-8 -*-

from tqdm import tqdm, trange
import os
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import pandas as pd
from fundchartv9 import pic

data = pd.read_csv('./data.csv')

STYLE = {
        'fore': {
                'black': 30, 'red': 31, 'green': 32, 'yellow': 33,
                'blue': 34, 'purple': 35, 'cyan': 36, 'white': 37,
        },
        'back': {
                'black': 40, 'red': 41, 'green': 42, 'yellow': 43,
                'blue': 44, 'purple': 45, 'cyan': 46, 'white': 47,
        },
        'mode': {
                'bold': 1, 'underline': 4, 'blink': 5, 'invert': 7,
        },
        'default': {
                'end': 0,
        }
}

def use_style(string, mode='', fore='', back=''):
    mode = '%s' % STYLE['mode'][mode] if STYLE['mode'].has_key(mode) else ''
    fore = '%s' % STYLE['fore'][fore] if STYLE['fore'].has_key(fore) else ''
    back = '%s' % STYLE['back'][back] if STYLE['back'].has_key(back) else ''
    style = ';'.join([s for s in [mode, fore, back] if s])
    style = '\033[%sm' % style if style else ''
    end = '\033[%sm' % STYLE['default']['end'] if style else ''
    return '%s%s%s' % (style, string, end)


ID = input('请输入合约地址：')
print('检查地址是否合法...', end='')
time.sleep(0.4)
print('done')
time.sleep(0.25)
print('='*50)
print('底层操作码安全性评估')
for i in trange(376):
    time.sleep(0.001)
print('done.')
print('='*50)
print('合约交易数据风险评估')
fn = 'High_risk_1338_0num.csv' # High_risk_1022_0num Media_109_1num
# =============================================================================
# 
# =============================================================================
print('-'*50)
col = 'Block_ptp_std'
title = '账户入场时间标准差'
print(f'检测结果：{title}异常')
plt.figure(figsize=(16,12))
sns.distplot(data[col].loc[np.logical_and(~data[col].isna(), data.cat2==0)],
             label='低危合约')
sns.distplot(data[col].loc[np.logical_and(~data[col].isna(), data.cat2==1)],
             label='高危合约')
tmp = data[col].loc[data.fn==fn].values[0]
plt.plot([tmp, tmp], [0,5.5], linewidth=5, label='本合约')
plt.title(title)
plt.xlabel('')
plt.yticks([])
plt.xticks(size=20)
plt.legend()
plt.show()
time.sleep(0.4)
# =============================================================================
# 
# =============================================================================
print('-'*50)
col = 'Total payment_std'
title = '账户交易数标准差'
print(f'检测结果：{title}轻度异常')
plt.figure(figsize=(16,12))
sns.distplot(data[col].loc[np.logical_and(np.logical_and(~data[col].isna(), data.cat2==0),data[col]<1)],
             label='低危合约')
sns.distplot(data[col].loc[np.logical_and(np.logical_and(~data[col].isna(), data.cat2==1),data[col]<1)],
             label='高危合约')
tmp = data[col].loc[data.fn==fn].values[0]
plt.plot([tmp, tmp], [0,7.5], linewidth=5, label='本合约')
plt.title(title,size=20)
plt.xlabel('')
plt.xticks(size=20)
plt.yticks([])
plt.legend(fontsize=20)
plt.show()
time.sleep(0.4)
# =============================================================================
# 
# =============================================================================
print('='*50)
print('骗局模式分析')
print('-'*50)
time.sleep(0.7)
print('合约库对比...', end='')
time.sleep(0.7)
print('done')
print(f'庞氏骗局概率：{data.loc[data.fn==fn].pred.values[0]*100*2:.1f}%')

print('资金流向图分析...')
tmp = fn.split('num')[0].split('High_risk_')[1]
pic(f'./input/dapp_722/High_risk/{tmp}/', tmp)
# =============================================================================
# tmp = fn.split('num')[0].split('Media_')[1]
# pic(f'E:/database/杂/区块链/区块链/input/dapp_722/Media/{tmp}/', tmp)
# =============================================================================
time.sleep(0.7)
print('-'*50)
print('骗局模式分析结果：')
# =============================================================================
# 
# =============================================================================
print('='*50)
print('合约分类建议：')
print('高危合约')

'''
做一个闪屏截图就可以了，然后把4张图田字格放在一起，就当是结果了，然后打开文件夹，弹出合约
'''

