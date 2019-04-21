import sys
pwd = sys.path[0]+'/'
import os
import fnmatch

file = open(pwd+'datalist.txt','w')

names = os.listdir(pwd+'../HockeyFights')

class1 = fnmatch.filter(names,'no*')
for name in class1:
	file.write(name+' 1\n')

class2 = fnmatch.filter(names,'fi*')
for name in class2:
	file.write(name+' 2\n')

file.close()

with open(pwd+'classInd.txt','w') as file:
	file.write('1 NonViolence\n2 Violence')