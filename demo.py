# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:43:36 2019

@author: SY
"""
import numpy as np
import pandas as pd
prob = [0.1002, 0.059, 0.0878, 0.1808, 0.2742, 0.2634, 0.1861, 0.2473, 0.1749, 0.2035,
 0.2715, 0.2715, 0.2715, 0.4745, 0.2715, 0.4163, 0.4745, 0.4745, 0.4061,
 0.4628, 0.464, 0.5344, 0.4061, 0.464, 0.464, 0.464, 0.464, 0.5344, 0.33,
 0.4628, 0.2862, 0.5513, 0.464, 0.464, 0.464, 0.5427, 0.3665, 0.4141, 0.4951,
 0.3277, 0.5035, 0.5513, 0.5218, 0.3226, 0.5839, 0.4402, 0.4402, 0.4402, 0.2158,
 0.4305, 0.4438, 0.4402, 0.2271, 0.4697, 0.2606, 0.2271, 0.1515, 0.4438,
 0.3763, 0.4019, 0.3763, 0.4053, 0.3023, 0.3763, 0.2902, 0.4305, 0.5026, 0.4927,
 0.4927, 0.4927, 0.3521, 0.3377, 0.5026, 0.3521, 0.4927, 0.3135, 0.3135,
 0.4927, 0.3482]

MAX = max(prob)
MIN = min(prob)

def FirstOrderLag(inputs,a):
	tmpnum = inputs[0]							#上一次滤波结果
	for index,tmp in enumerate(inputs):
		inputs[index] = (1-a)*tmp + a*tmpnum
		tmpnum = tmp
	return inputs


def ShakeOff(inputs,N):
	usenum = inputs[0]								#有效值
	i = 0 											#标记计数器
	for index,tmp in enumerate(inputs):
		if tmp != usenum:					
			i = i + 1
			if i >= N:
				i = 0
				inputs[index] = usenum
	return inputs



def WeightBackstepAverage(inputs):
	weight = np.array(range(1,np.shape(inputs)[0]+1))			#权值列表
	weight = weight/weight.sum()
 
	for index,tmp in enumerate(inputs):
		inputs[index] = inputs[index]*weight[index]
	return inputs

prob = WeightBackstepAverage(np.array(prob))
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)
prob = FirstOrderLag(np.array(prob), 0.5)

prob = (prob-prob.min())/prob.ptp() * (MAX-MIN) + MIN
prob = (prob-prob.min())/prob.ptp()*0.8 + MIN
# =============================================================================
# prob = (np.array(prob) 
#         + np.array(prob[1:]+[prob[-1]])
#         + np.array(prob[2:]+prob[-2:])) / 3
# 
# =============================================================================
df2 = pd.read_csv(f'./input/1.7/{cat}/{new_fn}.csv').rename(
            columns={'From': 'Personname'})
df2 = df2.sort_values(by='Block').reset_index(drop=True)
people_lst = []
prob_ts = []
for ix, people in enumerate(df2.Personname.values):
    if people not in people_lst:
        people_lst.append(people)
        prob_ts.append(df2.Block[ix])

def pic(posi,name):
    chartdir=posi+name+'.png'
    chartdir = chartdir.split('/')[-1]
    
    l=len(name)
    cate=posi[:-l-2]
    l=cate.rfind('/')
    cate=cate[l+1:]

    noflow=True  #无资金流动
    dicfrom = dict()
    flownum=0
    dicto = dict()
    dicres = dict()
    col = []
    data = []

    index = 0
    df_old = pd.read_csv(posi+name+".csv")
    #升序保证 从旧到新
    df=df_old.sort_values(by='Block',ascending=True)
    # dic for key:[block,value]
    flownum=0
    namelist = dict()
    ind=0
    maxvalue = 0
    totvalue=0
    for row in df.itertuples(index=True, name='Pandas'):
       if (getattr(row, "Flag") + getattr(row, "Internal")) == 1 :
            # col = df_old[index ] #提取一行      [外部协议flag1为正确，内部协议flag0为正确!]
            flownum+=1
            if float(getattr(row, "Value")) <= 0.00001:
                continue
            data = []
            totvalue+=getattr(row, "Value")
            if float(getattr(row, "Value"))>maxvalue:
                maxvalue=float(getattr(row, "Value"))
            data.append(getattr(row, "Block"))
            data.append(getattr(row, "Value"))

            # 定位为支出者 利用from
            fr=getattr(row, "From")
            tr=getattr(row, "To")
            #接触先后作为namelist的顺序
            if(fr not in namelist.keys()):
                ind+=1
                namelist[fr]=ind
            if (tr not in namelist.keys()):
                ind += 1
                namelist[tr] = ind
            blo=getattr(row,"Block")

            if fr in dicfrom:
                dicfrom[fr].append(data)
            else:
                dicfrom[fr] = []
                dicfrom[fr].append(data)
            # 定位为收入者 利用to
            if tr in dicto:
                dicto[tr].append(data)
            else:
                dicto[tr] = []
                dicto[tr].append(data)
# deal with
    blockf = []
    valuef = []
    personf = []
    blockt = []
    valuet = []
    persont = []
    num = 0
    valuemaxt = 0
    valuemaxf = 0
    l = []

    lf = []
    lt = []
    psnum=[]

    psname=[]
    psfrom=[]
    psto=[]
    bmin = 10000000
    bmax = 0
    # 有效者的交易总数统计
    tranin = [] #投资 作为from
    tranpay = []#回报 作为to
    #对人进行遍历
    totfrom = 0
    totto = 0
    blockmin = 10000000
    blockmax = 0

    for key in namelist.keys():
        try:
            totfrom = 0
            totto = 0
            # temporary list
            psname.append(key)
            psnum.append(namelist[key])

            if (key in dicfrom):
                lf = dicfrom[key]
                tranin.append(len(lf))  # 作为from 即投资
                for index in range(len(lf)):
                    block = int(lf[index][0])
                    if (str(lf[index][1]).find(",") != -1):
                        lf[index][1] = lf[index][1].replace(",", "")
                    if (math.isnan(float(lf[index][1]))):
                        value = 0
                    else:
                        value = float(lf[index][1])
                    totfrom += value
                    if (value != 0):
                        if block > bmax:
                            bmax = block
                        if block < bmin:
                            bmin = block
                        if(value>5):
                            large=False
                    blockf.append(block)
                    if maxvalue>4 or totvalue>30:
                        valuef.append(getsize(value))
                    else:
                        valuef.append(getsize_s(value))
                    personf.append(namelist[key])

                psfrom.append(totfrom)
            else:
                psfrom.append(0)
                tranin.append(0)
            if key in dicto:
                lt = dicto[key]
                tranpay.append(len(lt)) #作为to 即回报

                for index in range(len(lt)):
                    block = int(lt[index][0])
                    if (str(lt[index][1]).find(",") != -1):
                        lt[index][1] = lt[index][1].replace(",", "")
                    if (math.isnan(float(lt[index][1]))):
                        value = 0
                    else:
                        value = float(lt[index][1])
                    totto +=value
                    if(value!=0):
                        if block > bmax:
                            bmax = block
                        if block < bmin:
                            bmin = block
                        if (value > 5):
                            large = False
                    blockt.append(block)
                    if maxvalue>4 or totvalue>30:
                        valuet.append(getsize(value))
                    else:
                        valuet.append(getsize_s(value))
                    persont.append(namelist[key])
                psto.append(totto)
            else:
                psto.append(0)
                tranpay.append(0)


        except:
            print("error in",key)
    #df_num = pd.DataFrame({"Personname": psname, "Number": psnum,"Total investment":psfrom,"Total payment":psto,"Investment transnum":tranin,"Payment transnum":tranpay})
    #df_num.to_csv(posi+name+"num.csv", index=False, sep=",")
    plt.figure(figsize=(12,16))
    ax1 = plt.subplot(2,1,1)
    
    # 支出 from：蓝色
    # 设置标题
    plt.title('The Ether Flow Graph', size=20)
    if(ind==0):
        plt.yticks([0])
    else:
        if (ind<=8):
            plt.yticks(range(0, ind , 1))
        else:
            if (ind/5-int(ind/5)<0.25):
                plt.yticks(list(range(1, ind-1, int((ind) / 5)))+[ind])
            else:
                plt.yticks(list(range(1, int(ind/5)*5, int((ind) / 5))) + [ind])
    
    # 设置Y轴标签
    plt.ylabel('Person number', size=20)
    plt.xlim(bmin-3000, bmax)
# =============================================================================
#     plt.xticks(range(bmin - 10, bmax, int((bmax - bmin) / 6)), rotation=20)
# =============================================================================

    # 画散点图
    plt.scatter(blockf, personf, c='r', s=valuef, marker='o', label='investment', 
                linewidths=3)
    plt.scatter(blockt, persont, c='b', s=valuet, marker='o', label='payment', 
                linewidths=3)
    # 设置图标
    plt.legend(fontsize=20, markerscale=5)
    # 显示所画的图坐标

# =============================================================================
#     plt.savefig(chartdir, dpi=600)
# =============================================================================
    ax2 = plt.subplot(2,1,2,sharex=ax1)
    x = list(prob_ts[1:80])
    y = list(prob)
    x += [max(x)+10000, int((max(x)+bmax)/2),bmax-10000]
    y += [y[-1]*1.005, y[-1]*1.01, y[-1]*1.02]
    plt.plot(x, y, 'k', linewidth=2.5)
    plt.plot([bmin-3000, bmax-10000], [0.7,0.7], '#FF0000', 
             linewidth=3, label='high risk')
    plt.plot([bmin-3000, bmax-10000], [0.5,0.5], '#FF8073', 
             linewidth=3, label='low risk', alpha=0.5)
    
    plt.xlim(bmin-3000, bmax)
    plt.ylim(0, 1)
    
# =============================================================================
#     plt.fill_between([bmin-3000, bmax],[0.7,0.7],[1,1], 
#                      facecolor='#FF1800', label='high risk')
#     plt.fill_between([bmin-3000, bmax],[0.5,0.5],[0.7,0.7], 
#                      facecolor='#FF8073', label='middle risk')
#     plt.fill_between([bmin-3000, bmax],[0,0],[0.5,0.5], facecolor='yellow', 
#                      alpha=0.03, label='low risk')
# =============================================================================
    plt.legend(fontsize=20)
    plt.xlabel('Time stamp', size=20)
    plt.ylabel('Prediction', size=20)
    plt.show()
    
if __name__ == '__main__':
    pic(f'E:/database/杂/区块链/区块链/input/dapp_722/High_risk/{tmp}/', tmp)