from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import argparse

parser = argparse.ArgumentParser(description="Generate synthetic Compact FaceResNet W4A8 weights and embedding references")
parser.add_argument("output", type=Path)
out = parser.parse_args().output
out.mkdir(parents=True, exist_ok=True)
layers = [(3,16,3,1),(16,16,3,1),(16,16,3,1),
          (16,32,3,2),(32,32,3,1),(16,32,1,2),
          (32,64,3,2),(64,64,3,1),(32,64,1,2),(64,128,1,1)]
weights = []
shifts = []
nonzero = 0
total = 0
with (out/'weights.bin').open('wb') as f:
 for i,(ci,co,k,stride) in enumerate(layers):
    # Reconstruct the dense FP32 fixture in TestFaceResNet.cpp in Q12 units.
    w=np.where(np.arange(co*ci*k*k).reshape(co,ci,k,k)%2==0,1,-1)
    w[0,0,k//2,k//2]=4096 if i==0 or k==1 else 1024
    w[1,0 if i==0 else 1,k//2,k//2]=-4096 if i==0 else (4096 if k==1 else 1024)
    peak=np.abs(w).max(axis=(1,2,3))
    shift=np.floor(np.log2(7*4096/peak)).astype(int)
    q=np.rint(w*(2.0**shift[:,None,None,None])/4096).astype(np.int8)
    assert q.max()<=7 and q.min()>=-8
    shifts.append(shift);weights.append(q.astype(np.int64))
    q.tofile(f);nonzero+=np.count_nonzero(q);total+=q.size
def conv(idx, interval, stochastic, clip, relu=False):
    w=weights[idx]; ci,co,k,stride=layers[idx]
    def patches(x):
        a=np.pad(x,((0,0),(k//2,k//2),(k//2,k//2)))
        return sliding_window_view(a,(k,k),axis=(1,2))[:,::stride,::stride].transpose(0,3,4,1,2).reshape(ci*k*k,-1)
    l,h=map(patches,interval)
    wl,wh=np.minimum(w,0).reshape(co,-1),np.maximum(w,0).reshape(co,-1)
    a,b=wh@l+wl@h,wh@h+wl@l
    assert a.min()>=-2**19 and b.max()<2**19, (idx,a.min(),b.max())
    div=(1<<shifts[idx])[:,None]
    a=a//div;b=-((-b)//div) if stochastic and relu else b//div
    z=interval[0].shape[-1];s=(z+2*(k//2)-k)//stride+1
    if stochastic and relu:assert b.max()<=255
    return tuple(np.clip(x,*clip).reshape(co,s,s) for x in (a,b))
def run(sign,light):
    x=np.zeros((3,64,64),dtype=np.int64);x[0]=64*sign
    x=conv(0,(x,x),light,(0,255),True)
    for first,shortcut in [(1,None),(3,5),(6,8)]:
        act=conv(first,x,light,(0,255),True)
        main=conv(first+1,act,False,(-128,127))
        skip=x if shortcut is None else conv(shortcut,x,False,(-128,127))
        x=tuple(np.clip(a+b,0,255) for a,b in zip(main,skip))
        if light:assert (main[1]+skip[1]).max()<=255
    avg=tuple(a.sum(axis=(1,2),keepdims=True)//256 for a in x)
    return tuple(e.flatten()*64 for e in conv(9,avg,False,(-64,64)))
for sign in (1,-1):
    for light in (0,1):
        lo,hi=run(sign,light)
        np.stack([lo,hi]).astype('<i8').tofile(out/f'ref-{sign}-{light}.bin')
        print(sign,light,lo[:3],hi[:3], 'max interval (Q12)',(hi-lo).max())
print('W4 nonzero',nonzero,'of',total)
