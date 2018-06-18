from scipy.misc import imread
import os
import numpy as np
import sys

files = sorted(os.listdir(sys.argv[1]))


exemplars = set(i.split(',')[1].strip() for i in file('exemplars.txt').read().strip().split('\n'))
tuning = set(i.split(',')[1].strip() for i in file('tuning.txt').read().strip().split('\n'))
testing = set(i.split(',')[1].strip() for i in file('testing.txt').read().strip().split('\n'))

#Filter out extended tuning and testing data
nfiles = []
filter1 = []
filter2 = []
for i in files:
    if i.endswith('png'):
        continue
    ipart = i.split('_')[:-1]
    flagged = False
    j = []
    for p in ipart:
        j.append(p)
        jjoin='_'.join(j)
        if ((jjoin in tuning) or (jjoin in testing)):# and j!=ipart:
            flagged = True
            if jjoin in tuning:
                filter1.append(i)
            if jjoin in testing:
                filter2.append(i)
            #print "FILTERED",ipart
            break
    if not flagged:
        nfiles.append(i)
#print filter1, filter2, len(nfiles)
files = nfiles
#GET IMAGE SIZE
inf = imread(os.path.join(sys.argv[1],files[0]))
SIZE = 64
inputs = np.zeros((len(files)+len(filter1)+len(filter2),3,SIZE,SIZE),np.uint8)
outputs = np.zeros((len(files)+len(filter1)+len(filter2),1,SIZE,SIZE),np.uint8)

idx=0
eid=len(filter1)+len(filter2)
uid=0
tid=len(filter1)
print "TRAIN, %s-%s"%(eid,len(files)+len(filter1)+len(filter2))
print "TUNE, 0-%s"%tid
print "TEST, %s-%s"%(tid,eid)
print inputs.shape,outputs.shape
for i in filter1+filter2+files:
    name='_'.join(i.split('_')[:-1])
    inf = imread(os.path.join(sys.argv[1],name+'_in.jpg'))
    ouf = imread(os.path.join(sys.argv[1],name+'_out.png'))
    name='_'.join(i.split('_')[:-1])

    if name in tuning:
        idx = uid
        uid += 1
    elif name in testing:
        idx = tid
        tid += 1
    else:
        idx = eid
        eid += 1
    inputs[idx]=inf.transpose(2,0,1)
    outputs[idx,0]=ouf[:,:]
print uid,tid,eid,inputs.shape
np.savez(sys.argv[1]+'.npz',input=inputs,output=outputs)

