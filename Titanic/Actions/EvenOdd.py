import numpy as np
from sklearn import svm
clf = svm.SVC()
x=[1,2,3,4,5]+[y for y in range(14,100)]
xx=[]
for a in x:xx.append([a])
ax=np.array(xx)
ay=np.mod(ax[:,0],2)
print ay
	
clf.fit(ax,ay)
atrain=np.array([[9],[12],[13]])
print clf.predict(atrain)
for i in range(1,100):
	y=[i]
	print y,clf.predict(y)
