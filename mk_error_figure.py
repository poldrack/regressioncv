"""
make a figure showing the mean correlation
at each level of n

"""
import numpy as N
import pickle
import matplotlib.pyplot as plt

nvals=[32,48,64,96,128,256,512,1024]
#nvals=[32,48,64,96,128,256,512] #,512,1024]
cvtypes=['splithalf', 'balcv_hi', 'balcv_lo','loo']

meancorr=N.zeros((len(nvals),len(cvtypes)))

for n in range(len(nvals)):
    f=open('random_data/loosim_random_data_%d_subs.pkl'%nvals[n],'rb')
    d=pickle.load(f)
    f.close()
    for cv in range(len(cvtypes)):
        meancorr[n,cv]=N.mean(d[cvtypes[cv]])


plt.plot(nvals,meancorr[:,0])
plt.hold(True)
plt.plot(nvals,meancorr[:,1])
plt.plot(nvals,meancorr[:,2])
plt.plot(nvals,meancorr[:,3])
plt.legend(cvtypes,loc=4)
plt.xlabel("number of observations")
plt.ylabel('Mean correlation (predicted,actual)')
plt.savefig('corr_figure_random.png',format='png')

