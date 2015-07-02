
# The first thing to do is to import the relevant packages
# that I will need for my script, 
# these include the Numpy (for maths and arrays)
# and csv for reading and writing csv files
# If i want to use something from this I need to call 
# csv.[function] or np.[function] first

import csv as csv 
import numpy as np
import pandas as pd
import time as time
import operator
from operator import itemgetter
# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('../csv/train.csv', 'rb')) 
header = csv_file_object.next()  # The next() command just skips the 
                                 # first line which is a header
TrainData=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    TrainData.append(row)             # adding each row to the data variable
TrainData = np.array(TrainData)            # Then convert from a list to an array
                                 # Be aware that each item is currently
                                 # a string in this format

csv_file_object = csv.reader(open('../csv/test.csv', 'rb')) 
header = csv_file_object.next()  # The next() command just skips the 
                                 # first line which is a header
TestData=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    TestData.append(row)             # adding each row to the data variable
TestData = np.array(TestData)            # Then convert from a list to an array
# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.



iris=[]
target=[]
count_Train=0
InvalidData=[]

NormalTitle=['Mr','Mrs','Miss','Ms','Mme','Mlle']
RoyalTitle=['Master','Jonkheer','the Countness','Don','Sir','Lady']
MilitaryTitle=['Col','Major','Capt']
ReligousTitle=['Rev']
PhDTitle=['Dr']

Dict_CabinLevel={'A':[0,0], 'B':[0,0], 'C':[0,0], 'D':[0,0], 'E':[0,0], 'F':[0,0],'G':[0,0], 'T':[0,0]}
Dict_CabinRoom={}
Dict_Name={}
Dict_Ticket={}
###PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
###0	      ,1	,2,    3    4   5   6     7     8      9    10    11
for i in TrainData:
	#Name Dict
	tar=i[1].astype(np.int)
	namesplit= i[3].split(', ')
	y= namesplit[0]
	if(Dict_Name.get(y)==None):
		Dict_Name[y]=[1,0]
		Dict_Name[y].append(i)
	else: 
		Dict_Name[y][0]=Dict_Name[y][0]+1
		Dict_Name[y].append(i)
	if(tar==1): Dict_Name[y][1]=Dict_Name[y][1]+1
	#Cabin Room Dict
	if(i[10]!=''):
		x=i[10][0:4]
		if(Dict_CabinRoom.get(x)==None): 
			Dict_CabinRoom[x]=[1,0]
			Dict_CabinRoom[x].append(i)
		else:
			Dict_CabinRoom[x][0]=Dict_CabinRoom[x][0]+1
			Dict_CabinRoom[x].append(i)
		if(tar==1):Dict_CabinRoom[x][1]=Dict_CabinRoom[x][1]+1
	#Ticket Dict
	if(i[8]!=''):
		x=i[8]
		if(Dict_Ticket.get(x)==None):
			Dict_Ticket[x]=[1,0]
			Dict_Ticket[x].append(i)
		else:
			Dict_Ticket[x][0]=Dict_Ticket[x][0]+1
			Dict_Ticket[x].append(i)
		if(tar==1):Dict_Ticket[x][1]=Dict_Ticket[x][1]+1

print "Finish IDing"
#Print Ticket Dict:
Ticket_Array=[]
Ticket_Strange=[]
for i in Dict_Ticket:
	x=Dict_Ticket[i]
	if(ord(i[0])>=49 and ord(i[0])<=57): 
		Ticket_Array.append([i.astype(np.int),Dict_Ticket[i]])
	else:
		Ticket_Strange.append([i,Dict_Ticket[i]])

Ticket_Array=sorted(Ticket_Array, key=itemgetter(0))
#for i in Ticket_Array:
#	print i[0],i[1][2][9]
#	if(i[1][0]>1):
#		input_var = input("Enter something: ")
#		print i
print "ticket_array size=", len(Ticket_Array)
#time.sleep(5)
Ticket_Strange=sorted(Ticket_Strange, key=itemgetter(0))
#for i in Ticket_Strange:
#	if(i[1][0]>1):
#		input_var = input("Enter something: ")
#		print i
print "ticket_strange size=", len(Ticket_Strange)
#time.sleep(10)

#Print Name Dict
#for i in Dict_Name:
#	if(Dict_Name[i][0]>1):print i, Dict_Name[i]

#time.sleep(10)
#Print CabinRoom Dict
#for i in Dict_CabinRoom:
#	if(i[0]=='F' and Dict_CabinRoom[i][0]>1):
#		if(Dict_CabinRoom[i][0]==Dict_CabinRoom[i][1]):print i,Dict_CabinRoom[i]
#		elif(Dict_CabinRoom[i][1]==0): print i, Dict_CabinRoom[i]

#time.sleep(10)

count_valid=0
for i in TrainData:
	count_Train=count_Train+1
#	if((i.tolist()).count('')==0):
	tar=i[1].astype(np.int)
	iri=[0]*15
	#Pclass
	iri[0]=i[2].astype(np.int)
	#Name
	namesplit= i[3].split(', ')
	x= namesplit[1].split('.')[0]
	if(NormalTitle.count(x)==1):iri[10]=0
	elif(RoyalTitle.count(x)==1):iri[10]=1
	elif(MilitaryTitle.count(x)==1):iri[10]=1
	elif(ReligousTitle.count(x)==1):iri[10]=-1
	elif(PhDTitle.count(x)==1):iri[10]=0
#		y= namesplit[0]
	#Sex
	if(i[4]=="female"):iri[2]=0
	elif(i[4]=="male"):iri[2]=1
	#Age
	if(i[5]==''): 
		iri[3]=-100
		if(x=='Master'):iri[3]=1
	else:
		x=int((i[5]).astype(np.float))
		if(x<=12): 
			iri[3]=1
		else: 
			iri[3]=0
	#Sib
	iri[4]=(i[6]).astype(np.int)
	#Parch
	iri[5]=(i[7]).astype(np.int)
	#Number Of Company
	x=i[8]
	iri[14]=Dict_Ticket[x][0]
	#Ticket
#	iri[6]=i[8]
#	iri[6]=ord(i[8][0])
	#Fare
#	if(i[9]!=''):iri[7]=(i[9]).astype(np.float)
#	else:iri[7]=-100
	#Cabin
#	if(i[10]!=''):
#		x=i[10].replace(i[10][0],"")
#		iri[8]=ord(i[10][0])-64
#		if(x.split(' ')[0]!=''):
#	#		iri[11]= (int(x.split(' ')[0]))/2*2
#			iri[13]=(int(x.split(' ')[0]))%2
#		else: 
#	#		iri[11]=73/2*2
#			iri[13]=73%2
#	else:
#		iri[8]=-1000
#		iri[11]=-1000
	#Embark
#	if(i[11]=='S'):iri[9]=1
#	elif(i[11]=='C'):iri[9]=2
#	elif(i[11]=='Q'):iri[9]=0
#	count_valid=count_valid+1
	target.append(tar)
	iris.append(iri)

#print Dict_Cabin
#for i in Dict_Name:
#	print i, Dict_Name[i]
#time.sleep(4)
#

#print "valid data=",count_valid
print "Train Data=",count_Train, "number of input=", len(TrainData)
from sklearn import datasets
from sklearn import svm
print("load sklearn")
#clf=svm.SVC()
clf=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.00, kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=0.0001, verbose=False)

tarsize=np.size(target)

X,y=iris[50:tarsize-1], target[50:tarsize-1]
clf.fit(X,y)
Xsize=np.size(X)
Ysize=np.size(y)
#print Xsize, Ysize
time.sleep(5)
count =0
for i in range(0, Ysize-1):
	if(clf.predict(X[i])!=[y[i]]):
		print X[i], clf.predict(X[i]), y[i]
		print TrainData[i]
		count=count+1

print count
print Ysize
print count_Train

tarsize=np.size(target)
X,y=iris[0:49], target[0:49]
Xsize=np.size(X)
Ysize=np.size(y)
count=0
for i in range(0,Ysize):
	if(clf.predict(X[i])!=[y[i]]):
		print X[i], clf.predict(X[i]), y[i]
		count=count+1


print count
print Ysize

