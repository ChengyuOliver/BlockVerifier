from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import math
#生成资金流量图 （并生成用户资金流动表格（解除屏蔽特定的几行））（函数fundchargv9）
def getsize(v):
    s=float(0)
    if (v < 0.1):
        return v * 10
    if(v<0.5):
        return v*5+0.5
    if(v<1):
        return 3*v+1.5
    if(v<=10):
        return 0.6*v+4.5
    if(v>10 and v<100):
        s=v/20+10
    elif(v>=100 and v<1000):
        s=v/100+15
    elif (v>=1000 and v<10000):
        s=v/1000+25
    elif (v>=10000): #new add
        s=v/10000+35
    return s

def getsize_s(v):
    s=float(0)
    if(v<0.001):
        return v*8000
    if(v<0.01):
        return v*800+3
    if (v < 0.1):
        return v * 50+11
    if(v<0.5):
        return v*20+12
    if(v<1):
        return 10*v+17
    if(v<5):
        return 3*v+19
    if(v<=10):
        return v+24
    if(v<=100):
        return v/20+29
    if(v<1000):
        return  v/200+43
    if(v<10000):
        return v/1000+47
    elif (v>=10000): #new add
        return v/10000+56
    return v/100000+65

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

    #chart
    fig = plt.figure(figsize=(16,9))
    # 支出 from：蓝色
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('The Ether Flow Graph', size=20) #  of '+name+"("+cate+")"
    # 设置X轴标签
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
    plt.xlabel('Time stamp', size=20)
    # 设置Y轴标签
    plt.ylabel('Person number', size=20)
    plt.xlim(bmin-10, bmax)
    plt.xticks(range(bmin - 10, bmax, int((bmax - bmin) / 6)), rotation=20)

    # 画散点图
    ax1.scatter(blockf, personf, c='r', s=valuef, marker='o', label='investment')
    ax1.scatter(blockt, persont, c='b', s=valuet, marker='o', label='payment')
    # 设置图标
    plt.legend(fontsize=20, markerscale=5)
    # 显示所画的图坐标

# =============================================================================
#     plt.savefig(chartdir, dpi=600)
# =============================================================================
    plt.show()
    #plt.show()
    #该输出的直接在同目录输出
    plt.clf()
    plt.close("all")


    return flownum#不重要

if __name__ == '__main__':
    pic('C:/Users/Chengyu/Desktop/blockverifier-main/input/dapp_722/High_risk/1022_0/', '1022_0') # 951_0 1092_0