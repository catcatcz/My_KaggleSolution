print(__doc__)

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import finance
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold
from scipy import interpolate
from pydoc import help
from scipy.stats.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages

df_training = pd.read_csv('../csv/training_small.csv')
#df_testing  = pd.read_csv('../csv/testing.csv')
#hd_testing = pd.HDFStore('../csv/testing.csv')
#print "df_training=",df_training.describe()
#print "df_training.type=",df_training.dtypes
#print "df_testing =",df_testing.describe()
def GetTime(x):
	HO,MI,SE=x.split(':')
	HO=float(HO)
        MI=float(MI)
        SE=float(SE)
        PT=HO*3600+MI*60+SE
	return PT


def GatherSpread(x):
        i0=7
        imax=204
	SList=[[0,0] for i in range(50)]
        for i in range(50):
		PT=GetTime(x[i0+4*i+1])
		t1=x[i0+4*i+2]
		t2=x[i0+4*i+3]
		t1=float(t1)
		t2=float(t2)
		Sp=t1-t2
		SList[i]=[PT,Sp]
	return SList

def GetRate(x):
	RList=[[0,0] for i in range(49)]
	for i in range(49):
		RList[i]=[x[i],1.0/(x[i+1]-x[i])]
	RList=np.array(RList)
	return RList

SP=df_training.apply(GatherSpread,axis=1)

for i in range(len(SP)):
	temp=np.array(SP.iloc[i])
	RList=GetRate(temp)
	X,Y=RList[:,0],RList[:,1]
	plt.plot(X,Y,marker='.',linestyle='--')
PTS=PdfPages('ArrivialRates.pdf')
plt.xlim(28800,29000)
plt.yscale('log')
plt.xlabel('Time Point of Liquidity Shock/seconds')
plt.ylabel('Arrival Rates of Shocks/1/seconds')
plt.savefig(PTS,format='pdf')
PTS.close()

raw_input("xxxxx")


print "finish aply"
for i in range(len(SP)):
	temp=np.array(SP.iloc[i])
	X,Y=temp[:,0],temp[:,1]
	if(Y.min()<-100):
		print i,Y
		raw_input("xxx")
	#X,Y=SP.iloc[i][:,0],SP.iloc[i][:,1]
	plt.plot(X,Y,marker='.', linestyle='--')
PTS=PdfPages("Spreads.pdf")
plt.ylim(-5,1)
plt.xlim(28800,29000)
plt.xlabel('Time Point of Liquidity Shock/seconds')
plt.ylabel('SpreadValues/pence')
plt.savefig(PTS,format='pdf')
PTS.close()


raw_input("xxxx")



def GatherTime(x):
        i0=7
        imax=204
        QList=[[0,0] for i in  range(50)]
        TList=[[0,0] for i in  range(50)]
        for i in range(50):
		PT=GetTime(x[i0+4*i+1])
                if(x[i0+4*i]=='Q'): 
                        QList[i]=[PT,x[i0+4*i+3]]
                        TList[i]=None
                else:
                        TList[i]=[PT,x[i0+4*i+2]]
                        QList[i]=None
        QList=filter(None,QList)
        TList=filter(None,TList)
        return QList,TList


TS=df_training.apply(GatherTime,axis=1)
raw_input("xx")


QLAll=[[0,0] for i in range(len(TS)*50)]
TLAll=[[0,0] for i in range(len(TS)*50)]
QLsize=0
TLsize=0
for i in range(len(TS)):
	QL,TL=np.array(TS.iloc[i][0]),np.array(TS.iloc[i][1])
	QLAll[QLsize: QLsize+len(QL)]=QL
	TLAll[TLsize: TLsize+len(TL)]=TL
	QLsize=QLsize+len(QL)
	TLsize=TLsize+len(TL)
QLAll=np.array(QLAll)
TLAll=np.array(TLAll)	

"""
X,Y=QLAll[:,0],QLAll[:,1]
for i in range(len(QLAll)):
	if(Y[i]<1000):print X[i],Y[i]
plt.plot(X,Y)
plt.xlim(25000,40000)
plt.ylim(2350,2450)
plt.show()
raw_input("xxxx")
"""

def InterpAll(x):
	X,Y=x[:,0],x[:,1]
	f=interpolate.interp1d(X,Y)
	return f

PTS=PdfPages('LiquidityShockTimes_demean.pdf')
PTSzoom=PdfPages('LiquidityShockTimeszoom_demean.pdf')
fQ=InterpAll(QLAll)
fT=InterpAll(TLAll)
for i in range(len(TS)):
	QL,TL=np.array(TS.iloc[i][0]),np.array(TS.iloc[i][1])
	X,Y=TL[:,0],fT(TL[:,1])
	plt.plot(X,Y,color='r')
	x,y=QL[:,0],fQ(QL[:,1])
	plt.plot(x,y,color='g')
#plt.ylim(2300,2440)
plt.ylabel('Prices at Liquidity Shock/pence')
plt.xlabel('Time Point of Liquidity Shock/seconds')
plt.show()
#plt.savefig(PTS,format='pdf')
#PTS.close()
#plt.ylim(2340,2350)
#plt.xlim(45000,55000)
#plt.savefig(PTSzoom,format='pdf')
#PTSzoom.close()
#plt.show()
#x=df_training.iloc[0]
#QList,TList=GatherTime(x)
#TList=np.array(TList)
#X,Y=TList[:,0],TList[:,1]
#plt.plot(X,Y)
#plt.show()

