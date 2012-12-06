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

if len(sys.argv)>1:
    nsubs=int(sys.argv[1])
else:
    nsubs=48
    print 'no nsubs specified - assuming %d'%nsubs

# number of simulation runs - 1000 should be enough
if len(sys.argv)>2:
    nruns=int(sys.argv[2])
else:
    nruns=1000

# set this identifier to specify the nature of the
# simulations - it gets included in the output filename

identifier='random'


def get_sample_balcv(x,y,nfolds,pthresh=0.9):
    """
    This function uses anova across CV folds to find
    a set of folds that are balanced in their distriutions
    of the X value - see Kohavi, 1995
    """

    nsubs=len(x)

    # cycle through until we find a split that is good enough
    
    good_split=0
    while good_split==0:
        cv=cross_validation.KFold(n=nsubs,n_folds=nfolds,shuffle=True)
        ctr=0
        idx=N.zeros((nsubs,nfolds)) # this is the design matrix
        for train,test in cv:
            idx[test,ctr]=1
            ctr+=1

        lm=OLS(x-N.mean(x),idx).fit()

        if lm.f_pvalue>pthresh:
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

# save all the various corrs into a single dict to make saving easier

corrs={'splithalf':N.zeros(nruns),'loo':N.zeros(nruns),'balcv_lo':N.zeros(nruns),'balcv_hi':N.zeros(nruns)}

print 'running for %d subs'%nsubs
for run in range(nruns):
    # create the X distribution - this will be doubled over
    x=N.random.rand(nsubs/2).reshape((nsubs/2,1))

    # create two completely random Y samples
    y=N.random.rand(len(x)).reshape(x.shape)
    y2=N.random.rand(len(x)).reshape(x.shape)
    
    pred_y=N.zeros(x.shape)
    pred_y2=N.zeros(x.shape)

    # compute out-of-sample predictions for each Y dataset
    # i.e balanced split-half crossvalidation 
    lr=LinearRegression()
    lr.fit(x,y)
    pred_y2=lr.predict(x)
    lr.fit(x,y2)
    pred_y=lr.predict(x)

    pred_splithalf=N.vstack((pred_y,pred_y2))
    y_all=N.vstack((y,y2))
    corrs['splithalf'][run]=N.corrcoef(pred_splithalf[:,0],y_all[:,0])[0,1]


##     # this is an alternate way to implement split-half, removed for now
##     # after confirming that it gives identical results to the one above
##     split=cross_validation.KFold(n=len(x_all),n_folds=2,shuffle=False)
##     pred_splitcv=N.zeros(x_all.shape)
##     for train,test in split:
##         lr=LinearRegression()
##         lr.fit(x_all[train,:],y_all[train,:])
##         pred_splitcv[test]=lr.predict(x_all[test])
##    corrs['splitcv'][run]=N.corrcoef(pred_splitcv[:,0],y_all[:,0])[0,1]

    # this stacked version is needed for loo
    
    x_all=N.vstack((x,x))

    # do leave-one-out CV using sklearn tools
    loo=cross_validation.LeaveOneOut(nsubs)
    pred_loo=N.zeros(x_all.shape)
    for train,test in loo:
        lr=LinearRegression()
        lr.fit(x_all[train,:],y_all[train,:])
        pred_loo[test]=lr.predict(x_all[test])
    corrs['loo'][run]=N.corrcoef(pred_loo[:,0],y_all[:,0])[0,1]

    # get results with balanced CV for high and low threshold
    corrs['balcv_lo'][run]=get_sample_balcv(x_all,y_all,8,pthresh=0.001)
    corrs['balcv_hi'][run]=get_sample_balcv(x_all,y_all,8,pthresh=0.99)


f=open('loosim_%s_data_%d_subs.pkl'%(identifier,nsubs),'wb')
pickle.dump(corrs,f)
f.close()

print 'Mean correlation (predicted,true):"
for k in corrs.iterkeys():
     print k,N.mean(corrs[k])


