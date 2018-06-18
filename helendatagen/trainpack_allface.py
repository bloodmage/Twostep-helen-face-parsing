from scipy.misc import imread,fromimage
import PIL.Image as Image
import os
import numpy as np
import sys
from zipfile import ZipFile
from cStringIO import StringIO
zf = ZipFile('allface.zip','r')
zfnames = [i.split('/')[1] for i in zf.namelist() if i[-1]!='/']
files = sorted(zfnames)


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

inputs = np.zeros((len(files),3,100,100),np.uint8)
outputs = np.zeros((len(files),27),'f')

idx=0
eid=len(tuning)+len(testing)
uid=0
tid=len(tuning)
print "TRAIN, %s-%s"%(eid,len(files))
print "TUNE, 0-%s"%tid
print "TEST, %s-%s"%(tid,eid)

count = 0
for i in files:
    name='_'.join(i.split('_')[:-1])
    judge='_'.join(i.split('_')[:-2])
    
    inf = fromimage(Image.open(StringIO(zf.read('allface/%s_in.jpg'%name))))

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
    d=np.array([map(float,i.split(' ')) for i in zf.read('allface/%s.txt'%name).strip().split('\n')])
    outputs[idx]=d[:-1].flatten()
    count += 1
    if (count % 1000) == 0:
        print count,count*100/len(files)

print uid,tid,eid,inputs.shape,outputs.shape
np.savez('allface.npz',input=inputs,output=outputs)
