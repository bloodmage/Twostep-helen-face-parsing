from scipy.misc import imread
import os
import numpy as np
import sys

files = sorted(os.listdir('hardtask789'))


exemplars = set(i.split(',')[1].strip() for i in file('exemplars.txt').read().strip().split('\n'))
tuning = set(i.split(',')[1].strip() for i in file('tuning.txt').read().strip().split('\n'))
testing = set(i.split(',')[1].strip() for i in file('testing.txt').read().strip().split('\n'))

#Filter out extended tuning and testing data
nfiles = []
for i in files:
    ipart = i.split('_')
    if ipart[-1]!='in.jpg':
        continue
    ipart = ipart[:-1]
    flagged = False
    j = []
    jjoin='_'.join(ipart[:-1])
    if ((jjoin in tuning) or (jjoin in testing)) and ipart[-1]!='0':
            flagged = True
            #print "FILTERED",ipart
    if not flagged:
        nfiles.append(i)
files = nfiles

inputs = np.zeros((len(files),3,80,80),np.uint8)
outputs = np.zeros((len(files),3,80,80),np.uint8)
print tuning,testing
idx=0
eid=len(tuning)+len(testing)
uid=0
tid=len(tuning)
print "TRAIN, %s-%s"%(eid,len(files))
print "TUNE, 0-%s"%tid
print "TEST, %s-%s"%(tid,eid)
for i in files:
    name='_'.join(i.split('_')[:-1])
    judge='_'.join(i.split('_')[:-1])
    inf = imread(os.path.join('hardtask789',name+'_in.jpg'))

    if judge in tuning:
        idx = uid
        uid += 1
        #print '*',idx,i
    elif judge in testing:
        idx = tid
        tid += 1
        #print '!',idx,i
    else:
        idx = eid
        eid += 1
        #print '+',idx,i
    inputs[idx]=inf.transpose(2,0,1)
    for j in range(3):
        ouf = imread(os.path.join('hardtask789',name+'_0%s.png'%(j+7)))
        outputs[idx,j]=ouf[:,:]

print uid,tid,eid,inputs.shape,outputs.shape
np.savez('hardtask789.npz',input=inputs,output=outputs)
