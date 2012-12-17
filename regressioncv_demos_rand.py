"""
regressioncv_2d.py - code to simulate the effects of different
crossvalidation schemes on regression predictions

in this particular version, we create two sets of y values associated
with the same X values.  this allows us to examine the effect of using
split-half CV with exactly matched X distributions

Russ Poldrack, 2012
"""

import numpy as N
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from statsmodels.regression.linear_model import OLS
import sys
import pickle
from scipy.stats import scoreatpercentile

# set this identifier to specify the nature of the
# simulations - it gets included in the output filename

identifier='demosrand'


def get_sample_balcv(x,y,nfolds,pthresh=0.8):
    """
    This function uses anova across CV folds to find
    a set of folds that are balanced in their distriutions
    of the X value - see Kohavi, 1995
    """

    nsubs=len(x)

    # cycle through until we find a split that is good enough
    
    good_split=0
    while good_split==0:
        cv=cross_validation.KFold(n=nsubs,k=nfolds,shuffle=True)
        ctr=0
        idx=N.zeros((nsubs,nfolds)) # this is the design matrix
        for train,test in cv:
            idx[test,ctr]=1
            ctr+=1

        lm_x=OLS(x-N.mean(x),idx).fit()
        lm_y=OLS(y-N.mean(y),idx).fit()

        if lm_x.f_pvalue>pthresh and lm_y.f_pvalue>pthresh:
            good_split=1

    # do some reshaping needed for the sklearn linear regression function
    x=x.reshape((nsubs,1))
    y=y.reshape((nsubs,1))
    
    pred=N.zeros((nsubs,1))
    
    for train,test in cv:
        lr=LinearRegression()
        lr.fit(x[train,:],y[train,:])
        pred[test]=lr.predict(x[test])

    return N.corrcoef(pred[:,0],y[:,0])[0,1]


# data digitized from Demos et al. paper
data=N.loadtxt('demos_data.txt')
nsubs=data.shape[0]
x_all=data[:,0].reshape((nsubs,1))
y_all=data[:,1].reshape((nsubs,1))
nsubs=len(x_all)

nruns=1000

# save all the various corrs into a single dict to make saving easier

corrs={'splithalf':N.zeros(nruns),'loo':N.zeros(nruns),'balcv_lo':N.zeros(nruns),'balcv_hi':N.zeros(nruns)}

for run in range(nruns):
    N.random.shuffle(x_all)
    # do leave-one-out CV using sklearn tools
    loo=cross_validation.LeaveOneOut(nsubs)
    pred_loo=N.zeros(x_all.shape)
    for train,test in loo:
        lr=LinearRegression()
        lr.fit(x_all[train,:],y_all[train,:])
        pred_loo[test]=lr.predict(x_all[test])
    corrs['loo'][run]=N.corrcoef(pred_loo[:,0],y_all[:,0])[0,1]

    # get results with balanced CV for high and low threshold
    corrs['balcv_lo'][run]=get_sample_balcv(x_all,y_all,4,pthresh=0.001)
    corrs['balcv_hi'][run]=get_sample_balcv(x_all,y_all,4,pthresh=0.9)


f=open('loosim_%s_data_%d_subs.pkl'%(identifier,nsubs),'wb')
pickle.dump(corrs,f)
f.close()

print 'Mean correlation (predicted,true):'
for k in corrs.iterkeys():
     print k,N.mean(corrs[k])



