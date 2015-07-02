# The first thing to do is to import the relevant packages
# that I will need for my script, 
# these include the Numpy (for maths and arrays)
# and csv for reading and writing csv files
# If i want to use something from this I need to call 
# csv.[function] or np.[function] first

import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('../csv/train.csv', 'rb')) 
header = csv_file_object.next()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format

# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

women_only_stats = data[0::,4] == "female" # This finds where all 
                                           # the elements in the gender 
men_only_stats = data[0::,4] == "male"   # This finds where all the 
                                           # elements do not equal 
                                           # female (i.e. male)

# Using the index from above we select the females and males separately
women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

# Then we finds the proportions of them that survived
proportion_women_survived = \
                       np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = \
                       np.sum(men_onboard) / np.size(men_onboard) 

# and then print it out
print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

"""
test_file = open('../csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

prediction_file = open("genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()
"""
s1=(data[0::,4]=="female") & (data[0::,2]=="1")
women_class1=data[s1,1].astype(np.float)
s2=(data[0::,4]=="female") & (data[0::,2]=="2")
women_class2=data[s2,1].astype(np.float)
s3=(data[0::,4]=="female") & (data[0::,2]=="3")
women_class3=data[s3,1].astype(np.float)

women_class1_percent = np.sum(women_class1)/np.size(women_class1)
women_class2_percent = np.sum(women_class2)/np.size(women_class2)
women_class3_percent = np.sum(women_class3)/np.size(women_class3)
print 'women_class1_percent= %s' % women_class1_percent
print 'women_class2_percent= %s' % women_class2_percent
print 'women_class3_percent= %s' % women_class3_percent

s1=(data[0::,4]=="male") & (data[0::,2]=="1")
men_class1=data[s1,1].astype(np.float)
s2=(data[0::,4]=="male") & (data[0::,2]=="2")
men_class2=data[s2,1].astype(np.float)
s3=(data[0::,4]=="male") & (data[0::,2]=="3")
men_class3=data[s3,1].astype(np.float)

men_class1_percent = np.sum(men_class1)/np.size(men_class1)
print 'men_class1_percent= %s' % men_class1_percent
men_class2_percent = np.sum(men_class2)/np.size(men_class2)
print 'men_class2_percent= %s' % men_class2_percent
men_class3_percent = np.sum(men_class3)/np.size(men_class3)
print 'men_class3_percent= %s' % men_class3_percent


FamilySizeSet=[]
for i in data:
	FamilySize=i[6].astype(np.int)+i[7].astype(np.int)+1
	FamilySizeSet.append(FamilySize)

Family_size_ceiling=max(FamilySizeSet)
print Family_size_ceiling
Family_size_floor=min(FamilySizeSet)
print Family_size_floor

number_of_classes = len(np.unique(data[0::,2])) 
number_of_FamilySizes =Family_size_ceiling-Family_size_floor+1

female_vector=[]
g="female"
for i in xrange(number_of_classes):
    for j in xrange(number_of_FamilySizes):
	 mask=(data[0::,4]==g) & (data[0::,2].astype(np.float)==i) & (data[0::,9].astype(np.float)< j+1) & (data[0::,9].astype(np.float)>=j)
	 temp = data[mask,1].astype(np.float)
	 print "np size= %s" % np.size(temp)
	 percent=np.sum(temp)/np.size(temp)
	 ss={i,j,percent}
	 female_vector.append(ss)
