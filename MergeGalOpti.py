#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def Merge(prefix,start,end):
    MergedOpti,val1,val2,val3=np.load('{}{}.npy'.format(prefix,start))
    for i in range(start+1,end+1):
        MergedOpti=np.concatenate((MergedOpti,np.load('{}{}.npy'.format(prefix,i))[0]),axis=1)
    np.save('{}merged.npy',(MergedOpti,val1,val2,val3))


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("prefix", help = "Prefix to use.",type = str)
    parser.add_argument("end", help = "ending point.",type = int)
    parser.add_argument("--start", help = "starting point.",type = int,default=0)
    args = parser.parse_args()
    Merge(args.prefix,args.start,args.end)
