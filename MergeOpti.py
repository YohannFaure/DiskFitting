#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
If you need to merge optimizations files générated with ModelingEmcee, with the option --resume, just use that.

BEWARE, it's recursive (I wanted to have fun) and might use some memory.
'''
import numpy as np

def mergetworesults(a,b):
    return(np.concatenate((a,b),axis=1))

def merger_rec(l):
    if len(l)==1:
        samples,_,_,_ =np.load(l[0])
        return(samples)
    elif len(l)==2:
        a,_,_,_ =np.load(l[0])
        b,_,_,_ =np.load(l[1])
        return(mergetworesults(a,b))
    else :
        a=len(l)//2
        return( mergetworesults(merger_rec(l[:a]),merger_rec(l[a:])) )

def merger(l):
    samples,_,_,_ =np.load(l[0])
    for i in range(1,len(l)):
        aaa,_,_,_=np.load(l[i])
        samples = mergetworesults(samples,aaa)
    return(samples)

l=['results/optimization/opti_37_300_5000.npy',
'results/optimization/opti_37_300_1000part2.npy',
'results/optimization/opti_37_300_5000part3.npy',
'results/optimization/opti_37_300_1000part4.npy',
'results/optimization/opti_37_300_1000part5.npy',
'results/optimization/opti_37_300_1000part6.npy',
'results/optimization/opti_37_300_1000part7.npy',
'results/optimization/opti_37_300_1000part8.npy',
'results/optimization/opti_37_300_1000part9.npy',
'results/optimization/opti_37_300_1000part10.npy',
'results/optimization/opti_37_300_1000part11.npy',
'results/optimization/opti_37_300_1000part12.npy',
'results/optimization/opti_37_300_1000part13.npy',
'results/optimization/opti_37_300_1000part14.npy',
'results/optimization/opti_37_300_1000part15.npy',
'results/optimization/opti_37_300_1000part16.npy',
'results/optimization/opti_37_300_1000part17.npy',
'results/optimization/opti_37_300_1000part18.npy',
'results/optimization/opti_37_300_1000part19.npy',
'results/optimization/opti_37_300_1000part20.npy',
'results/optimization/opti_37_300_1000part21.npy',
'results/optimization/opti_37_300_1000part22.npy']



#xxx=merger_rec(l)
xxx=merger(l)

_,a,b,c=np.load(l[-1])
np.save('results/optimization/BigOpti.npy',(xxx,a,b,c))
