from cloudimp import cloudimport
cloudimport('https://raw.githubusercontent.com/bloodmage/libref/master/layerbase.py')
cloudimport('https://raw.githubusercontent.com/bloodmage/libref/master/fractallayer.py')
cloudimport('https://raw.githubusercontent.com/bloodmage/libref/master/hengelossvalid.py')

import numpy as np
import numpy.random as npr
import scipy.misc as misc

from fractallayer import ShrinkshapeFractal, ExpandshapeFractal, AggregationLayer, ConvKeepLayer
from layerbase import Model, SymbolDataLayer, safefile, DrawPatch, DrawMaskedPatch, LogSoftmaxLayer, LabelLoss, makesoftmaxlabel
from hengelossvalid import binaryloss_label

import theano
import theano.tensor as T
import sys
import PIL
import PIL.Image
import copy

MOMENTUM = 0.8
LEARN_RATE = 5e-7
BATCHSTEP = 100

def rndtransform(rng,inp):
    angle = rng.uniform(-15,15)
    scale = rng.uniform(0.9,1.1)
    dx = rng.uniform(-5,5)
    dy = rng.uniform(-5,5)
    
    def transform(single):
        simage = misc.toimage(single)
        timg = PIL.Image.new(simage.mode,simage.size)
        timg.paste(simage.resize((int(simage.size[0]*scale),int(simage.size[1]*scale)),PIL.Image.BICUBIC),(int(dx-simage.size[0]*(scale-1)*0.5),int(dy-simage.size[1]*(scale-1)*0.5)))
        simage = timg
        simage = simage.rotate(angle,PIL.Image.BICUBIC,expand=False)
        return misc.fromimage(simage)

    oup = np.empty(inp.shape,'f')
    for j in range(inp.shape[0]):
        for i in range(inp.shape[1]):
            oup[j,i]=transform(inp[j,i])
    return oup

if __name__=="__main__":
    rng = npr.RandomState(23455)
    rndstream = T.shared_randomstreams.RandomStreams(12345)
    data=np.load('hardtask4.npz')
    inp=data['input']
    oup=data['output']
    data = None
    inp1 = inp
    oup1 = oup


    data = np.load('hardtask5.npz')
    inp2 = data['input']
    oup2 = data['output']

    inp2 = inp2[:,:,:,::-1]
    oup2 = oup2[:,:,:,::-1]

    inp = np.empty((inp1.shape[0]*2,)+inp1.shape[1:],inp1.dtype)
    inp[::2]=inp1
    inp[1::2]=inp2

    oup = np.empty((oup1.shape[0]*2,)+oup1.shape[1:],oup1.dtype)
    oup[::2]=oup1
    oup[1::2]=oup2
    inp1=inp2=oup1=oup2=None
    data = None
    print inp.shape,oup.shape
    #oup=np.array(data['output'],'f')/255.0*2-1
    #oup=np.where(oup>50,np.float32(1.0),np.float32(-1.0))#/255.0*2-1
    OUT = oup.shape[1]
    #data=np.load('rndsample.npz')
    #inp2=np.array(data['input'],'f')
    #oup2=data['output']
    #inp2[:] = (inp2-inp2.mean(axis=(2,3),keepdims=True))/(inp2.std(axis=(2,3),keepdims=True))
    #oup2=np.array(data['output'],'f')/100.0
    #inp = np.vstack([inp,inp2])
    #oup = np.vstack([oup,oup2])
    
    #inp2=oup2=None
    
    if 0:
        oup2 = np.abs(oup)
        for i in range(1,oup.shape[1]):
            for j in range(1,oup.shape[1]):
                if i!=j:
                    oup[:,i]-=oup2[:,j]*0.1
        oup2 = None
        np.clip(oup,-1,1,out=oup)
        #HARD NEGATIVE
        oup-=1e-3
        oup = np.where(oup>0,oup,-1)

    #inp,oup = shuffle_in_unison_inplace(rng,inp,oup)

    #SAMPLE VALIDATION SET
    valid = inp[460:660]
    voutp = oup[460:660]
    valid = valid.astype('f')
    valid -= np.mean(valid,axis=(2,3),keepdims=True)
    valid /= valid.std(axis=(2,3),keepdims=True)
    voutp = makesoftmaxlabel(voutp)
    inp = inp[660:]
    oup = oup[660:]
    
    if len(sys.argv)>1:
        BS = 1
    else:
        BS = 10

    print "LOAD TO GPU"
    #l0 = DataLayer(inp, oup, 1)
    todropout = T.scalar('todropout')
    l0 = l1 = SymbolDataLayer(inp.shape, oup.shape, BS)
    
    #Split and shrink part
    l2 = ShrinkshapeFractal(l1)
    l4 = ShrinkshapeFractal(l2)
    l8 = ShrinkshapeFractal(l4)

    l1_2 = ConvKeepLayer(rng, l1, (8,3,9,9))
    l2_2 = ConvKeepLayer(rng, l2, (16,3,9,9))
    l4_2 = ConvKeepLayer(rng, l4, (24,3,9,9))
    l8_2 = ConvKeepLayer(rng, l8, (32,3,9,9))

    l1_2s = ShrinkshapeFractal(l1_2)
    l2_2s = ShrinkshapeFractal(l2_2)
    l4_2s = ShrinkshapeFractal(l4_2)
    l2_2e = ExpandshapeFractal(l2_2, l2)
    l4_2e = ExpandshapeFractal(l4_2, l4)
    l8_2e = ExpandshapeFractal(l8_2, l8)

    l1_2a = AggregationLayer(l1_2, l2_2e) #8+16
    l2_2a = AggregationLayer(l2_2, l1_2s, l4_2e) #8+16+24
    l4_2a = AggregationLayer(l4_2, l2_2s, l8_2e) #16+24+32
    l8_2a = AggregationLayer(l8_2, l4_2s) #24+32

    l1_3 = ConvKeepLayer(rng, l1_2a, (8,24+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l2_3 = ConvKeepLayer(rng, l2_2a, (16,48+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l4_3 = ConvKeepLayer(rng, l4_2a, (24,72+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l8_3 = ConvKeepLayer(rng, l8_2a, (32,56,5,5), dropout = todropout, dropoutrnd = rndstream)

    l1_3s = ShrinkshapeFractal(l1_3)
    l2_3s = ShrinkshapeFractal(l2_3)
    l4_3s = ShrinkshapeFractal(l4_3)
    l2_3e = ExpandshapeFractal(l2_3, l2)
    l4_3e = ExpandshapeFractal(l4_3, l4)
    l8_3e = ExpandshapeFractal(l8_3, l8)

    l1_3a = AggregationLayer(l1_3, l2_3e)
    l2_3a = AggregationLayer(l2_3, l1_3s, l4_3e)
    l4_3a = AggregationLayer(l4_3, l2_3s, l8_3e)
    l8_3a = AggregationLayer(l8_3, l4_3s)

    l1_4 = ConvKeepLayer(rng, l1_3a, (8,24+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l2_4 = ConvKeepLayer(rng, l2_3a, (16,48+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l4_4 = ConvKeepLayer(rng, l4_3a, (24,72+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l8_4 = ConvKeepLayer(rng, l8_3a, (32,56,5,5), dropout = todropout, dropoutrnd = rndstream)

    l1_4s = ShrinkshapeFractal(l1_4)
    l2_4s = ShrinkshapeFractal(l2_4)
    l4_4s = ShrinkshapeFractal(l4_4)
    l2_4e = ExpandshapeFractal(l2_4, l2)
    l4_4e = ExpandshapeFractal(l4_4, l4)
    l8_4e = ExpandshapeFractal(l8_4, l8)

    l1_4a = AggregationLayer(l1_4, l2_4e)
    l2_4a = AggregationLayer(l2_4, l1_4s, l4_4e)
    l4_4a = AggregationLayer(l4_4, l2_4s, l8_4e)
    l8_4a = AggregationLayer(l8_4, l4_4s)

    l1_5 = ConvKeepLayer(rng, l1_4a, (8,24+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l2_5 = ConvKeepLayer(rng, l2_4a, (16,48+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l4_5 = ConvKeepLayer(rng, l4_4a, (24,72+2,5,5), dropout = todropout, dropoutrnd = rndstream)
    l8_5 = ConvKeepLayer(rng, l8_4a, (32,56,5,5), dropout = todropout, dropoutrnd = rndstream)
    
    #Expand and propogate
    l8_5e = ExpandshapeFractal(l8_5, l8)
    l4_5a = AggregationLayer(l4_5, l8_5e) #24+32

    l4_6 = ConvKeepLayer(rng, l4_5a, (24,56+2,5,5), dropout = todropout, dropoutrnd = rndstream)

    l4_6e = ExpandshapeFractal(l4_6, l4)
    l2_5a = AggregationLayer(l2_5, l4_6e) #16+24

    l2_6 = ConvKeepLayer(rng, l2_5a, (16,40+2,5,5), dropout = todropout, dropoutrnd = rndstream)

    l2_6e = ExpandshapeFractal(l2_6, l2)
    l1_5a = AggregationLayer(l1_5, l2_6e) #8+16

    l1_6 = ConvKeepLayer(rng, l1_5a, (2*OUT+8,24+2,5,5))
    l1_7 = ConvKeepLayer(rng, l1_6, (OUT+1, 2*OUT+8, 9,9),Nonlinear=False)
    l1_7sm = LogSoftmaxLayer(l1_7)

    lout = LabelLoss(l1_7sm, l1)
    
    model = Model(
        l1,l2,l4,l8,
        l1_2,l2_2,l4_2,l8_2,
        l1_2s,l2_2s,l4_2s,
              l2_2e,l4_2e,l8_2e,
        l1_2a,l2_2a,l4_2a,l8_2a,
        l1_3,l2_3,l4_3,l8_3,
        l1_3s,l2_3s,l4_3s,
              l2_3e,l4_3e,l8_3e,
        l1_3a,l2_3a,l4_3a,l8_3a,
        l1_4,l2_4,l4_4,l8_4,
        l1_4s,l2_4s,l4_4s,
              l2_4e,l4_4e,l8_4e,
        l1_4a,l2_4a,l4_4a,l8_4a,
        l1_5,l2_5,l4_5,l8_5,
        l8_5e,l4_5a,
        l4_6,
        l4_6e,l2_5a,
        l2_6,
        l2_6e,l1_5a,
        l1_6,l1_7,l1_7sm,lout)
    a,b = T.fscalar(), T.fscalar()
    obinary,_tp,_fp,_tn,_fn,_F = binaryloss_label(l1_7.output, lout.output, 0, a,b)#4.6, 1.41)

    cost = lout.loss

    params = model.params()
    momentums = model.pmomentum()

    grads = T.grad(cost, params)
    updates = []
    updating = 0.0
    for grad, momentum in zip(grads, momentums):
        updates.append((momentum, MOMENTUM*momentum - LEARN_RATE*grad))
        updating = updating + T.sum(abs(momentum))

    for param, momentum in zip(params, momentums):
        updates.append((param, param + momentum))

    index = T.lscalar('index')
    print "COMPILE0"
    #train_model = theano.function(inputs=[l0.data, l0.label],
    #                              outputs=[cost,updating], updates=updates,
    #                              givens=[(todropout,np.float32(0.0))])
    print "COMPILE1"
    #valid_model = theano.function(inputs=[l0.data, l0.label], outputs=cost, on_unused_input='ignore', givens=[(todropout,np.float32(0.0))], no_default_updates=True)
    print "COMPILE2"
    vis_model = theano.function(inputs=[l0.data, l0.label],
                                outputs=model.outputs(),
                                on_unused_input='ignore',
                                givens=[(todropout,np.float32(0.0))])
    #use_model = theano.function(inputs=[l0.data], outputs=l1_7sm.output, on_unused_input='ignore', givens=[(todropout,np.float32(0.0))], no_default_updates=True)
    print "COMPILE3"
    vis_binary_model = theano.function(inputs=[l0.data, l0.label,a,b], outputs=model.outputs() + [obinary], on_unused_input='ignore', givens=[(todropout,np.float32(0.0))], no_default_updates=True)
    valid_binary_model = theano.function(inputs=[l0.data, l0.label,a,b], outputs=[_tp,_fp,_tn,_fn], on_unused_input='ignore', givens=[(todropout,np.float32(0.0))], no_default_updates=True)
    with safefile('modelf45_2') as loadf:
        if loadf:
            model.load(loadf.rb())

    if len(sys.argv)>1 and sys.argv[1] == 'test':
        print "Get precision & recall"
        def getF(ab):
            pr = []
            #valid = inp[230:330]
            #voutp = oup[230:330]
            a,b=ab.astype('f')
            for j in range(200/BS):
                #sys.stdout.write('.')
                #sys.stdout.flush()
                v=valid_binary_model(valid[j*BS:(j+1)*BS],voutp[j*BS:(j+1)*BS],a,b)
                resp = vis_binary_model(valid[j*BS:(j+1)*BS],voutp[j*BS:(j+1)*BS],a,b)
                layer = 0
                for i in resp:
                    layer += 1
                    if len(i.shape)!=4:
                        #PIL.Image.fromarray(np.array((i-np.min(i))/(np.max(i)-np.min(i))*255,np.uint8)).save('stepf45/%s_resp%s.png'%(j,layer))
                        continue
                    #PIL.Image.fromarray(DrawPatch(i[0:1])).save('stepf45/%s_resp%s.jpg'%(j,layer), quality=100)
                #PIL.Image.fromarray(DrawMaskedPatch(resp[0][0:1], resp[-1][0:1,0:1])).save('stepf45/maskresult_%s.jpg'%(j), quality=100)
                #PIL.Image.fromarray(DrawMaskedPatch(resp[0][0:1], resp[-2][0:1,0:1])).save('stepf45/masktruth_%s.jpg'%(j), quality=100)
                #v=v[:-1]
                print v
                pr.append(v)
            pr = np.array(pr).sum(axis=0)
            tp=pr[0]
            fp=pr[1]
            tn=pr[2]
            fn=pr[3]
            prec = tp/(tp+fp)
            reca = tp/(tp+fn)
            F = -2*prec*reca/(prec+reca)
            print ab,F
            return F[0]
        #print getF(np.array(( 6.31853293 , 8.53974314),'f'))
        from scipy.optimize import fmin
        print fmin(getF,np.array(( 6.31853293 , 8.53974314),'f')).xopt
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1] == 'serv':
        oups = oup[0:1]
        inp = data = oup = None
        valid = voutp = None
        import xmlrpclib
        from SimpleXMLRPCServer import SimpleXMLRPCServer

        def processarray(arraystr):
            import base64
            try:
                rawarray = base64.b64decode(arraystr)
                arr = np.array(np.fromstring(rawarray, dtype=np.uint8).reshape((1,3,64,64)),'f')
                b,c,d=np.copy(arr)[0]
                arr[0,:]=[d,c,b]
                np.save('p45.npy',arr)
                arr = (arr-arr.mean(axis=(2,3),keepdims=True))/(arr.std(axis=(2,3),keepdims=True)+1e-10)
                output = vis_model(arr,oups)[-3]
                np.save('p45out.npy',output)
                print output.shape
                return base64.b64encode(output.tostring())
            except:
                import traceback
                traceback.print_exc()
                return ""

        server = SimpleXMLRPCServer(('0.0.0.0',int(sys.argv[2])))
        server.register_function(processarray, 'processarray')
        server.serve_forever()

    b = npr.randint(l0.n_batches)
    LOSS0 = 1e100
    resultid = 0
    tol = 0
    d = 0
    for i in range(1000000):
        oupt=rndtransform(copy.deepcopy(rng),oup[b*BS:(b+1)*BS])
        inpt=rndtransform(rng,inp[b*BS:(b+1)*BS])
        oupt=makesoftmaxlabel(oupt)
        inpt[:]=(inpt - np.mean(inpt,axis=(2,3),keepdims=True))/(inpt.std(axis=(2,3),keepdims=True)+1e-10)
        d += [float(t) for t in train_model(inpt,oupt)][1]
        sys.stdout.write('.')
        sys.stdout.flush()
        if i % BATCHSTEP == BATCHSTEP-1:
            print d,"DRAW"
            d = 0
            #Draw model
            layer = 0
            for i in model.paramlayers():
                layer += 1
                param = i.params[0].get_value()
                if len(param.shape)!=4:
                    if hasattr(i,'reshape') and i.reshape!=None:
                        PIL.Image.fromarray(DrawPatch(param.reshape((-1,)+i.reshape[1:]))).save('stepf45/layer_%s.jpg'%layer, quality=100)
                    else:
                        PIL.Image.fromarray(np.array((param-np.min(param))/(np.max(param)-np.min(param))*255,np.uint8)).save('stepf45/layer_%s.png'%layer)
                    continue
                PIL.Image.fromarray(DrawPatch(param)).save('stepf45/layer_%s.jpg'%layer, quality=100)
            #Draw response
            resp = vis_model(inpt,oupt)
            layer = 0
            for i in resp:
                layer += 1
                if len(i.shape)!=4:
                    PIL.Image.fromarray(np.array((i-np.min(i))/(np.max(i)-np.min(i))*255,np.uint8)).save('stepf45/resp%s.png'%layer)
                    continue
                PIL.Image.fromarray(DrawPatch(i[0:1])).save('stepf45/resp%s.jpg'%layer, quality=100)
            #PIL.Image.fromarray(DrawMaskedPatch(resp[0][0:1], resp[-2][0:1])).save('steps5/maskresult_%s.jpg'%resultid, quality=100)
            #PIL.Image.fromarray(DrawMaskedPatch(resp[0][0:1], resp[-1][0:1])).save('steps5/masktruth_%s.jpg'%resultid, quality=100)
            #resultid+=1
            #Check validset
            if 1:
                LOSS1 = 0.0
                for j in range(460/BS):
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    LOSS1 += valid_model(valid[j*BS:(j+1)*BS],voutp[j*BS:(j+1)*BS])
                print LOSS1
                if LOSS1>LOSS0:
                    print "Converge on validset"
                    tol+=1
                    if tol>5:
                        sys.exit(0)
                else:
                    tol=0
                print "NEW LOSS",LOSS1
                LOSS0 = LOSS1
        b = npr.randint(l0.n_batches)
        #Save model
        with safefile('modelf45_2') as savef:
            model.save(savef.wb())

