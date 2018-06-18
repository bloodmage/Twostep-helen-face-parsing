import scipy.misc as misc
import os
import PIL.Image as img
import multiprocessing as mp
import random
import numpy as np
from PIL.ImageChops import lighter,invert

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

def PadImage(oimg,l,r,u,d):
    bg = img.new(oimg.mode,(oimg.size[0]+l+r,oimg.size[1]+u+d),0)
    bg.paste(oimg,(l,u))
    return bg

def ProcessImg(inp,l,r,u,d,size,rot,oup):
    if isinstance(inp,str):
        data = img.open(inp)
    else:
        data = inp
    data = PadImage(data,l,r,u,d)
    data = RoundResize(data,size)
    data.rotate(rot,img.BICUBIC,False).save(oup,quality=100)

def ProcessRatio(inp,l,r,u,d,size,rot,oup):
    if isinstance(inp,str):
        data = img.open(inp)
    else:
        data = inp
    data = PadImage(data,l,r,u,d)
    data = RoundResize(data,size)
    mat = np.array(misc.fromimage(data.rotate(rot,img.BICUBIC,False)),'f')
    xsum = np.sum(mat * np.array(range(100)).reshape((1,100)))
    ysum = np.sum(mat * np.array(range(100)).reshape((100,1)))
    asum = np.sum(mat)
    
    xsum /= asum*100
    ysum /= asum*100
    asum /= 100*100*255

    file(oup,'a+').write('%s %s %s\n'%(xsum,ysum,asum))

def dolabel(i):
    print i
    
    #Not modified image
    def ProcessSeries(l,r,u,d,size,rot,tag):
        ProcessImg(r'images\%s.jpg'%i,l,r,u,d,size,rot,r'allface\%s_%s_in.jpg'%(i,tag))
        for oid in range(2,10):
            ProcessImg(r'labels\%s\%s_lbl0%s.png'%(i,i,oid),l,r,u,d,size,rot,r'allface\%s_%s_0%s.png'%(i,tag,oid))
        return
        ProcessRatio(r'labels\%s\%s_lbl01.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        ProcessRatio(r'labels\%s\%s_lbl02.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        ProcessRatio(r'labels\%s\%s_lbl03.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        ProcessRatio(r'labels\%s\%s_lbl04.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        ProcessRatio(r'labels\%s\%s_lbl05.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        ProcessRatio(r'labels\%s\%s_lbl06.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        ProcessRatio(r'labels\%s\%s_lbl07.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        ProcessRatio(r'labels\%s\%s_lbl08.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        ProcessRatio(r'labels\%s\%s_lbl09.png'%(i,i),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
        #Combine hair and bg
        bg=img.open(r'labels\%s\%s_lbl00.png'%(i,i))
        hair=img.open(r'labels\%s\%s_lbl10.png'%(i,i))
        ProcessRatio(invert(lighter(bg,hair)),l,r,u,d,size,rot,r'allface\%s_%s.txt'%(i,tag))
    
    ProcessSeries(0,0,0,0,(64,64),0,0)
    return #In program argumentation
    #Data Argumentation

    for s in range(1,100):
        l=random.randrange(0,10)
        r=random.randrange(0,10)
        u=random.randrange(0,10)
        d=random.randrange(0,10)
        rot=random.uniform(-15,15)
        ProcessSeries(l,r,u,d,(64,64),rot,s)



if __name__=="__main__":
    labels = os.listdir(r'labels')
    try: os.mkdir('allface2')
    except: pass

    cnt = 0
    pool = mp.Pool(4)
    pool.map(dolabel,labels)
