import scipy.misc as misc
import os
import PIL.Image as img
import multiprocessing as mp
import random
import numpy as np

lbl = 2
RNDSIZE = 10
def ExtendCrop(oimg,bound):
    l,u,r,d = bound
    pl=pu=pr=pd=0
    if l<0: pl=-l
    if u<0: pu=-u
    if r>=oimg.size[0]: pr=r-oimg.size[0]+1
    if d>=oimg.size[1]: pd=d-oimg.size[1]+1
    #print (r-l+1,d-u+1),(l+pl,u+pu,r-pr,d-pd),(pl,pu)
    newimg = img.new(oimg.mode,(r-l+1,d-u+1),0)
    newimg.paste(oimg.crop((l+pl,u+pu,r-pr,d-pd)),(pl,pu))
    return newimg

def RoundResize(oimg,size):
    oimg.thumbnail(size,img.ANTIALIAS)
    bg = img.new(oimg.mode,size,0)
    bg.paste(oimg,((size[0]-oimg.size[0])/2, (size[1]-oimg.size[1])/2))
    return bg
def PasteCenter(oimg,size):
    bg = img.new(oimg.mode,size,0)
    bg.paste(oimg,((size[0]-oimg.size[0])/2, (size[1]-oimg.size[1])/2))
    return bg

def dolabel(i):
    print i
    component = misc.imread(os.path.join('labels',i,'%s_lbl02.png'%i))
    
    #Check component boundary
    l=u=1e10
    r=d=-1

    splate = np.where(component>0,1,0)
    if splate.flatten().sum()==0: return
    v = np.nonzero(splate.sum(axis = 0))[0]
    l,r = v[0], v[-1]
    v = np.nonzero(splate.sum(axis = 1))[0]
    u,d = v[0], v[-1]
    
    if r==-1: return
    #Simple sample: center and with fixed padding (to round)
    w=r-l+1
    h=d-u+1
    #print w,h
    l1=l-h/2-32
    r1=r+h/2+32
    u1=u-w/2-32
    d1=d+w/2+32
    #print r1-l1,d1-u1

    #ExtendCrop(img.open(r'images/%s.jpg'%i),(l1,u1,r1,d1)).resize((32,32),img.ANTIALIAS).save(r'simpletask/%s_in.jpg'%i,quality=100)
    #ExtendCrop(img.open(os.path.join('labels',i,'%s_lbl02.png'%i)),(l1,u1,r1,d1)).resize((32,32),img.ANTIALIAS).save(r'simpletask/%s_out.png'%i)

    #Hard task: argumenting 20 different scale and rotate settings

    PasteCenter(ExtendCrop(img.open(r'images\%s.jpg'%i),(l1,u1,r1,d1)),(64,64)).save(r'hardtask%s\%s_in.jpg'%(lbl,i),quality=100)
    PasteCenter(ExtendCrop(img.open(os.path.join('labels',i,'%s_lbl0%s.png'%(i,lbl))),(l1,u1,r1,d1)),(64,64)).save(r'hardtask%s\%s_out.png'%(lbl,i))
    return
    for s in range(20):
        l1=l-random.randrange(h/4,h/2+RNDSIZE)-32
        r1=r+random.randrange(h/4,h/2+RNDSIZE)+32
        u1=u-random.randrange(w/4,w/2+RNDSIZE)-32
        d1=d+random.randrange(w/4,w/2+RNDSIZE)+32
        rot=random.uniform(-15,15)

        PasteCenter(ExtendCrop(img.open(r'images/%s.jpg'%i),(l1,u1,r1,d1)),(64,64)).rotate(rot,img.BICUBIC,False).save(r'hardtask%s/%s_%s_in.jpg'%(lbl,i,s),quality=100)
        PasteCenter(ExtendCrop(img.open(os.path.join('labels',i,'%s_lbl0%s.png'%(i,lbl))),(l1,u1,r1,d1)),(64,64)).rotate(rot,img.BICUBIC,False).save(r'hardtask%s/%s_%s_out.png'%(lbl,i,s))



if __name__=="__main__":
    labels = os.listdir(r'labels')
    #try: os.mkdir('simpletask')
    #except: pass
    try: os.mkdir('hardtask%s'%lbl)
    except: pass

    cnt = 0
    pool = mp.Pool(4)
    pool.map(dolabel,labels)
