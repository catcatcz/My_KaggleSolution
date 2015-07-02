import pandas as pd
import numpy as np
import time as time

# For .read_csv, always use header=0 when you know row 0 is the header row
df_train = pd.read_csv('../csv/train.csv', header=0)
#print df_train.dtypes
#print df_train.describe()
df_test = pd.read_csv('../csv/test.csv', header=0)
df_test['Survived']=''
df_test=df_test[['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
#print df_test.dtypes
#print df_test.describe()
df_tot=df_train.append(df_test)
#print df_tot.dtypes
#print df_tot.describe()
df_tot['Gender']=4
df_tot['Gender'] = df_tot['Sex'].map( lambda x: x[0].upper() )
df_tot['Gender'] = df_tot['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df_tot.loc[df_tot['Fare'].isnull(),'Fare']=15
df_tot=df_tot.drop('Sex', axis=1)
df_tot['Company']=1
Ticket_Number_List=df_tot.Ticket.unique()
for Ticket in Ticket_Number_List:
	nb=len(df_tot[df_tot.Ticket==Ticket])
	df_tot.loc[(Ticket==df_tot.Ticket),'Company']=nb
df_tot['Ticket_Number']=0
df_tot['Ticket_Number']=df_tot['Ticket'].map(lambda x:x.split()[-1])
df_tot['Family_Name']=None
df_tot['Title']=None
df_tot['Family_Name']=df_tot['Name'].map(lambda x:x.split(',')[0])
df_tot['Title']=df_tot['Name'].map(lambda x:x.split(', ')[1].split('.')[0])
df_tot['Family_Size']=df_tot['SibSp']+df_tot['Parch']

df_tot['AgeGroup']=0
def CheckAge(x):
	if(x['Age']<12 or x['Title']=='Master'): return 1
	elif(x['Age']>45) : return -1
	else:return 0
df_tot['AgeGroup']=df_tot.apply(CheckAge, axis=1)


def CheckMarried(x):
	if(x['Title']=='Mrs' and x['Company']>1 and x['SibSp']>0):return 1
	elif(x['Title']=='Mr' and x['Company']>1 and x['SibSp']>0 and x['AgeGroup']!=1):
		y=df_tot[((df_tot.Family_Name==x['Family_Name']) & (df_tot.Title=='Mrs') )]
		if(len(y)>=1): return 1
	else: return 0
df_tot['Married']=df_tot.apply(CheckMarried, axis=1)

df_tot['ServeGod']=df_tot['Title'].map(lambda x:x=='Rev')

Family_Name_List=df_tot.Name.map(lambda x:x.split(',')[0]).unique()
Title_List=df_tot.Name.map(lambda x:x.split(', ')[1].split('.')[0]).unique()

GodServer_Family_Name_List=df_tot[df_tot.ServeGod==True].Family_Name.unique()


df_tot['SpouseDead']=0
def CheckSpouseDead(x):
	if((x['Title']=='Mrs') & (x['Family_Name'] in GodServer_Family_Name_List)):return 1
	if(x['Title']=='Mrs' and x['SibSp']>=1):
		spouse=df_tot[((df_tot.Family_Name==x['Family_Name']) & (df_tot.Ticket_Number==x['Ticket_Number']) & (df_tot.Title=='Mr') & (df_tot.SibSp>=1))]
		if( 0 in  spouse['Survived'].unique()): return 1
		else: return 0
	if(x['Title']=='Mr' and x['AgeGroup']!=1 and x['SibSp']>=1):
		spouse=df_tot[((df_tot.Family_Name==x['Family_Name']) & (df_tot.Ticket_Number==x['Ticket_Number']) & (df_tot.Title=='Mrs'))]
		if( 0 in  spouse['Survived'].unique()): return 1
		else: return 0
	return 0

df_tot['SpouseDead']=df_tot.apply(CheckSpouseDead,axis=1)		

df_tot['SiblingDead']=0
def CheckSiblingDead(x):
	if(x['AgeGroup']==1 and x['SibSp']>=1):
		family=df_tot[((df_tot.Family_Name==x['Family_Name']) & (df_tot.Ticket_Number==x['Ticket_Number']) &(df_tot.AgeGroup==1) )]
		if(0 in family['Survived'].unique()):return 1
	else:return 0

df_tot['SiblingDead']=df_tot.apply(CheckSiblingDead, axis=1)
#time.sleep(10)
#print df_tot.dtypes
#print df_tot.describe()

df_train_trim=df_tot[df_tot['Survived']!='']
df_train_trim=df_train_trim.drop(['PassengerId','Name','Age','Ticket','Cabin','Embarked','Title','Family_Name','Ticket_Number'],axis=1)
DataTrain=df_train_trim.as_matrix()
Target=DataTrain[:,0]
DataTrain=DataTrain[:,1:8]
from sklearn import svm
clf=svm.SVC()
from sklearn import tree
clf_DecisionTree=tree.DecisionTreeClassifier()

DataTrain_classify=[[],[],[],[],[],[]]
Target_classify=   [[],[],[],[],[],[]]
PID=		   [[],[],[],[],[],[]]
index =0
def SeparateClass(DataTrain, df_train,DataTrain_classify, Target_classify,PID,index,index0):
	for i in DataTrain:
		if(df_train.ix[index-index0]['Pclass']==1 and df_train.ix[index-index0]['Sex']=='female'):
			DataTrain_classify[0].append(i)	
			Target_classify[0].append(Target[index-index0])
			PID[0].append(index)
		elif(df_train.ix[index-index0]['Pclass']==2 and df_train.ix[index-index0]['Sex']=='female'):
			DataTrain_classify[1].append(i)	
			Target_classify[1].append(Target[index-index0])
			PID[1].append(index)
		elif(df_train.ix[index-index0]['Pclass']==3 and df_train.ix[index-index0]['Sex']=='female'):
			DataTrain_classify[2].append(i)	
			Target_classify[2].append(Target[index-index0])
			PID[2].append(index)
		elif(df_train.ix[index-index0]['Pclass']==1 and df_train.ix[index-index0]['Sex']=='male'):
			DataTrain_classify[3].append(i)	
			Target_classify[3].append(Target[index-index0])
			PID[3].append(index)
		elif(df_train.ix[index-index0]['Pclass']==2 and df_train.ix[index-index0]['Sex']=='male'):
			DataTrain_classify[4].append(i)	
			Target_classify[4].append(Target[index-index0])
			PID[4].append(index)
		elif(df_train.ix[index-index0]['Pclass']==3 and df_train.ix[index-index0]['Sex']=='male'):
			DataTrain_classify[5].append(i)	
			Target_classify[5].append(Target[index-index0])
			PID[5].append(index)
		index=index+1


SeparateClass(DataTrain, df_train,DataTrain_classify, Target_classify,PID,index,0)
count_classify=[0]*6
count_classify_DecisionTree=[0]*6
for i in range(0,6):
	X,y=DataTrain_classify[i],Target_classify[i]
	clf.fit(X,y)
	clf_DecisionTree.fit(X,y)
	Length=len(X)
	for j in range(0, Length):
		if(clf.predict(X[j])!=[y[j]]):
			#print df_train.ix[PID[i][j]]
			#print X[j], y[j],clf.predict(X[j]) 
			#var = raw_input("Please enter something: ")
			count_classify[i]=count_classify[i]+1
		if(clf_DecisionTree.predict(X[j])!=[y[j]]):
	#		print df_train.ix[PID[i][j]]
	#		print X[j], y[j],clf_DecisionTree.predict(X[j]) 
	#		var = raw_input("Please enter something: ")
			count_classify_DecisionTree[i]=count_classify_DecisionTree[i]+1
	print "wrong ", count_classify[i], "out of ", Length, "in class", i
	print "Decision Tree got wrong",count_classify_DecisionTree[i]
	#var = raw_input("Please enter something: ")
print index, "SVM got wrong", sum(count_classify), " Decision Got wrong", sum(count_classify_DecisionTree)
#print df_train_trim.dtypes
#print df_train_trim.describe()
#print Family_Name_List
#print Title_List
df_test_trim=df_tot[df_tot['Survived']=='']
df_test_trim=df_test_trim.drop(['PassengerId','Name','Age','Ticket','Cabin','Embarked','Title','Family_Name','Ticket_Number'],axis=1)
DataTest=df_test_trim.as_matrix()
Target=DataTest[:,0]
DataTest=DataTest[:,1:8]
DataTest_classify=	[[],[],[],[],[],[]]
TargetTest_classify=    [[],[],[],[],[],[]]
PID_Test=       	[[],[],[],[],[],[]]
index=891
Solution=[]
SeparateClass(DataTest, df_test,DataTest_classify, TargetTest_classify,PID_Test,index,891)
for i in range(0,6):
	X,y=DataTrain_classify[i], Target_classify[i]
	clf_DecisionTree.fit(X,y)
	X_test=DataTest_classify[i]
	Length=len(X_test)
	print Length
	for j in range(0, Length):
		#print X_test[j]
		Solution.append( [PID_Test[i][j]+1,clf_DecisionTree.predict(X_test[j])[0]])
from operator import itemgetter
Solution=sorted(Solution, key=itemgetter(0))
import csv
with open('Solution.csv','wb') as csvfile:
	spamwriter = csv.writer(csvfile)
	spamwriter.writerow('PassengerID,Survived')
	for i in Solution:
		spamwriter.writerow(i)
