import sys
pwd = sys.path[0]+'/'
import os
import fnmatch

test_split = 0.3

file1 = open(pwd+'testlist.txt','w')
file2 = open(pwd+'trainlist.txt','w')

names = os.listdir(pwd+'../HockeyFights')

class1 = fnmatch.filter(names,'no*')
k = int(len(class1)*test_split)
for name in class1[:k]:
	file1.write(name+' 1\n')
for name in class1[k:]:
	file2.write(name+' 1\n')

class2 = fnmatch.filter(names,'fi*')
k = int(len(class2)*test_split)
for name in class2[:k]:
	file1.write(name+' 2\n')
for name in class2[k:]:
	file2.write(name+' 2\n')

file1.close()
file2.close()

with open(pwd+'classInd.txt','w') as file:
	file.write('1 NonViolence\n2 Violence')