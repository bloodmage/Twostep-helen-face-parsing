from scipy.misc import imread,fromimage
import PIL.Image as Image
import os
import numpy as np
import sys
from zipfile import ZipFile
from cStringIO import StringIO
import itertools
zf = ZipFile('sets.zip','r')
zfnames = [i.split('/')[1] for i in zf.namelist() if i[-1]!='/']
files = set(['_'.join(i.split('_')[:-2]) for i in zfnames])
exemplars = [i for i in list(i.split(',')[1].strip() for i in file('exemplars.txt').read().strip().split('\n')) if i in files]
tuning = [i for i in list(i.split(',')[1].strip() for i in file('tuning.txt').read().strip().split('\n')) if i in files]
testing = [i for i in list(i.split(',')[1].strip() for i in file('testing.txt').read().strip().split('\n')) if i in files]
lens =  len(exemplars)+len(tuning)+len(testing)

maps = [
        (['_0_in.jpg'],['_0_out.png'],'2'),
        (['_1_in.jpg'],['_1_out.png'],'3'),
        (['_2_in.jpg'],['_2_out.png'],'4'),
        (['_3_in.jpg'],['_3_out.png'],'5'),
        (['_4_in.jpg'],['_4_out.png'],'6'),
        (['_5_in.jpg'],['_5_out.png','_6_out.png','_7_out.png'],'789'),
        ]


for infiles,oufiles,suffix in maps:
    inputs = np.zeros((len(files),3,300,300),np.uint8)
    outputs = np.zeros((len(files),len(oufiles),300,300),np.uint8)

    idx=0
    eid=len(tuning)+len(testing)
    uid=0
    tid=len(tuning)
    print "TRAIN, %s-%s"%(eid,len(files))
    print "TUNE, 0-%s"%tid
    print "TEST, %s-%s"%(tid,eid)

    count = 0
    for i in itertools.chain(tuning,testing,exemplars):
        name=i
        judge=i
        

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
        inf = fromimage(Image.open(StringIO(zf.read('sets/%s%s'%(name,infiles[0])))))
        inputs[idx]=inf.transpose(2,0,1)
        d=np.array([fromimage(Image.open(StringIO(zf.read('sets/%s%s'%(name,osuffix))))) for osuffix in oufiles])
        outputs[idx]=d
        count += 1
        if (count % 1000) == 0:
            print count,count*100/len(files)

    print uid,tid,eid,inputs.shape,outputs.shape
    np.savez('aligned%s.npz'%suffix,input=inputs,output=outputs)

