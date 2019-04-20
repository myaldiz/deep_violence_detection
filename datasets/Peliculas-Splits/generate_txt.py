import sys
pwd = sys.path[0]+'/'
import os

test_split = 0.3

file1 = open(pwd+'testlist.txt','w')
file2 = open(pwd+'trainlist.txt','w')

names = os.listdir(pwd+'../Peliculas/noFights')
k = int(len(names)*test_split)
for name in names[:k]:
	file1.write('noFights/'+name+' 1\n')
for name in names[k:]:
	file2.write('noFights/'+name+' 1\n')

names = os.listdir(pwd+'../Peliculas/fights')
k = int(len(names)*test_split)
for name in names[:k]:
	file1.write('fights/'+name+' 2\n')
for name in names[k:]:
	file2.write('fights/'+name+' 2\n')

file1.close()
file2.close()

with open(pwd+'classInd.txt','w') as file:
	file.write('1 NonViolence\n2 Violence')