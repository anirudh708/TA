'''
Created on Jan 22, 2015

@author: Gramener-pc
'''

import random
import time

i = {}
for x in range(0,2000000):
    i[x] = x+3

d=[]
for s in xrange(0,100000):
    d.append(random.randint(0,200000))

a = time.time()

subDict = dict(map(lambda k: (k, i.get(k, None)), d))
b=time.time()
subDict = {l : i[l] for l in d if l in i}

c=time.time()
print (b-a)
print (b-a)/(c-b)