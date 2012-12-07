
import numpy as N
import pickle
import matplotlib.pyplot as plt

nvals=[32,64,128,256,512,1024]
cvtypes=['splithalf', 'balcv_hi', 'loo', 'balcv_lo']

meancorr=N.zeros((len(nvals),len(cvtypes)))

for n in range(len(nvals)):
    f=open('loosim_random_data_%d_subs.pkl'%nvals[n],'rb')
    d=pickle.load(f)
    f.close()
    for cv in range(len(cvtypes)):
        meancorr[n,cv]=N.mean(d[cvtypes[cv]])


plt.plot(nvals,meancorr[:,0])
plt.hold(True)
plt.plot(nvals,meancorr[:,1])
plt.plot(nvals,meancorr[:,2])
plt.plot(nvals,meancorr[:,3])
plt.legend(cvtypes)