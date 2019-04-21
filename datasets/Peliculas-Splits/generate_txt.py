import sys
pwd = sys.path[0]+'/'
import os

file1 = open(pwd+'datalist.txt','w')

names = os.listdir(pwd+'../Peliculas/noFights')
for name in names:
	file1.write('noFights/'+name+' 1\n')

names = os.listdir(pwd+'../Peliculas/fights')
for name in names:
	file1.write('fights/'+name+' 2\n')

file1.close()

with open(pwd+'classInd.txt','w') as file:
	file.write('1 NonViolence\n2 Violence')