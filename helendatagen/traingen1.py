import scipy.misc as misc
import os
import PIL.Image as img
import multiprocessing as mp
import random
import numpy as np
lbl = 1

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
    component = misc.imread(os.path.join('labels',i,'%s_lbl01.png'%i))
    
    #Check component boundary
    l=u=1e10
    r=d=-1

    for y in range(component.shape[0]):
        for x in range(component.shape[1]):
            if component[y,x]>0:
                if x<l: l=x
                if y<u: u=y
                if x>r: r=x
                if y>d: d=y
    
    if r==-1: return
    #Simple sample: center and with fixed padding (to round)
    w=r-l+1
    h=d-u+1
    #print w,h
    l1=l-40
    r1=r+40
    u1=u-40
    d1=d+40

    print r1-l1,d1-u1

    #ExtendCrop(img.open(r'images/%s.jpg'%i),(l1,u1,r1,d1)).resize((32,32),img.ANTIALIAS).save(r'simpletask/%s_in.jpg'%i,quality=100)
    #ExtendCrop(img.open(os.path.join('labels',i,'%s_lbl02.png'%i)),(l1,u1,r1,d1)).resize((32,32),img.ANTIALIAS).save(r'simpletask/%s_out.png'%i)

    #Hard task: argumenting 20 different scale and rotate settings

    #PasteCenter(ExtendCrop(img.open(r'images\%s.jpg'%i),(l1,u1,r1,d1)),(64,64)).save(r'hardtask%s\%s_in.jpg'%(lbl,i),quality=100)
    #PasteCenter(ExtendCrop(img.open(os.path.join('labels',i,'%s_lbl0%s.png'%(i,lbl))),(l1,u1,r1,d1)),(64,64)).save(r'hardtask%s\%s_out.png'%(lbl,i))

    for s in range(100):
        while True:
            sa=random.randrange(l1,r1-60)-l1
            sb=random.randrange(u1,d1-60)-u1
            rot=random.uniform(-15,15)
            lblimg = ExtendCrop(img.open(os.path.join('labels',i,'%s_lbl0%s.png'%(i,lbl))),(l1,u1,r1,d1)).rotate(rot,img.BICUBIC,False).crop((sa,sb,sa+60,sb+60))
            if np.sum(misc.fromimage(lblimg))==0:
                continue
            ExtendCrop(img.open(r'images/%s.jpg'%i),(l1,u1,r1,d1)).rotate(rot,img.BICUBIC,False).crop((sa,sb,sa+60,sb+60)).save(r'hardtask%s/%s_%s_in.jpg'%(lbl,i,s),quality=100)
            lblimg.save(r'hardtask%s/%s_%s_out.png'%(lbl,i,s))
            break


if __name__=="__main__":
    labels = os.listdir(r'labels')
    #try: os.mkdir('simpletask')
    #except: pass
    try: os.mkdir('hardtask%s'%lbl)
    except: pass

    cnt = 0
    pool = mp.Pool(4)
    pool.map(dolabel,labels)
