import numpy as np
import pandas as pd
import math
import emcee
import numpy.random as rand
import scipy
from emcee import PTSampler
from scipy.stats import binom as binomial
from scipy.stats import gamma
from scipy.stats import multinomial
import matplotlib.pyplot as plt
import corner
from scipy.stats import norm
#from ipywidgets import FloatProgress
#from IPython.display import display
import sys
import pickle
#import PTMCMCSampler
#from PTMCMCSampler import PTMCMCSampler as ptmcmc
from scipy.stats import beta as betafunction

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
#Real data, read in with Pandas.
vobs_pandas = pd.read_csv('Xvax.csv')
vtrials_pandas = pd.read_csv('Nvax.csv')
sobs_pandas = pd.read_csv('Ysero.csv')
strials_pandas = pd.read_csv('Nsero.csv')
cases_pandas = pd.read_csv('reported_cases_all.csv')


#Select the locations you want to work with
#loc_subset = ['BANDUNDU','KINSHASA','KATANGA','EQUATEUR','MANIEMA','ORIENTALE']
loc_subset = ['EQUATEUR']
vobs = vobs_pandas.loc[:,loc_subset]
vtrial = vtrials_pandas.loc[:,loc_subset]
sobs = sobs_pandas.loc[:,loc_subset]
strial = strials_pandas.loc[:,loc_subset]
cases = cases_pandas.loc[:,loc_subset]

#Convert to numpy arrays
vo = vobs.values
vt = vtrial.values
so = sobs.values
st = strial.values
co = cases.values

vo = np.sum(vo,axis=-1)
vt = np.sum(vt,axis=-1)
so = np.sum(so,axis=-1)
st = np.sum(st,axis=-1)
co = np.sum(co,axis=-1)

total_cases = np.sum(co)

# Define Prior and Likelihood
#Now define the prior and likelihood:

def logl(x,vobs,sobs,vtry,stry,cobs): #The model to be extracted is pmeas_i * Data[i]
    lnlike = 0.
    N = len(vobs)
    
    #Parameters
    valpha = va = x[0]
    vbeta = vb = x[1]
    vheight = vh = x[2]
    falpha = fa =x[3]
    fbeta = fb = x[4]
    fheight = fh =x[5]
    veff = ve = x[6]
    seff = x[7]
        
    #Enforce prior
    lprior = lnprior(x)
    if lprior==-np.inf:
        return -np.inf  
    
    vprob = np.zeros(N)
    sprob = np.zeros(N)
    cprob = np.zeros(N)
    for i in range(0,N):
        vprob[i] = 1. - np.exp(-vheight*(1. - np.exp(-((float(i+1))/vbeta)**valpha)))
        sprob[i] = 1. - np.exp(-vheight*(veff)*(1. - np.exp(-((float(i+1))/vbeta)**valpha)) - 
                                          fheight*(1. - np.exp(-(float(i+1))/fbeta)**falpha))
        #Here's where the probability of a case is calculated:
        cprob[i] = np.exp(-(fheight*(falpha/fbeta) * (float(i+1)*12./fbeta)**(falpha-1.)*np.exp(-(float(i+1)*12./fbeta)**falpha)))\
            *(np.exp(-(veff)*vheight*(1.-np.exp(-(float(i+1)*12./vbeta)**valpha))-fheight*(1.-np.exp(-(float(i+1)*12./fbeta)**falpha)))) * np.exp(-0.03 * (float(i+1)/12))   
            
        if vprob[i] < 0.0:
            return -np.inf
        if sprob[i] < 0.0:
            return -np.inf
        lnlike += np.log(binomial.pmf(vobs[i],vtry[i],vprob[i]))
        lnlike += np.log(binomial.pmf(int(seff*sobs[i]),stry[i],sprob[i]))
   
    cprob /= sum(cprob) #I was using this normalization initially.
    lnlike += scipy.stats.multinomial.logpmf(cobs,total_cases,cprob)
    #lnlike += np.log(scipy.stats.poisson.pmf(cobs,cprob))
            
    return lnlike+lprior

def lnprior(x):
    logp = 0.0
    valpha = x[0]
    vbeta = x[1]
    vheight = x[2]
    falpha = x[3]
    fbeta = x[4]
    fheight = x[5]
    veff = x[6]
    #x[7] = 1.
    seff = x[7]
    
    #In case we want a hard limit on any of these parameters. 
    if valpha<0.0:
        return - np.inf
    if vbeta<0.0:
        return -np.inf
    if vheight<0.0:
        return -np.inf
    if falpha<0.0:
        return -np.inf
    if fbeta<0.0 or fbeta>1000.:
        return -np.inf
    if fheight<0.0:
        return -np.inf
    if veff<0.25 or veff>1.:
        return -np.inf
    if seff<0.75 or seff >1.5:
        return -np.inf
    
    logp += np.log(gamma.pdf(valpha,a=2.,scale=1.))
    #logp += np.log(gamma.pdf(falpha,a=2.,scale=1.))
    logp += np.log(gamma.pdf(vbeta,a=2.,scale=1.))
    #logp += np.log(gamma.pdf(fbeta,a=2.,scale=1.))
    logp += np.log(gamma.pdf(vheight,a=2.,scale=1.))
    #logp += np.log(gamma.pdf(fheight,a=2.,scale=1.))
    logp += np.log(scipy.stats.norm.pdf(veff,0.75,0.1))
    #logp += np.log(scipy.stats.cauchy.pdf(veff,0.75,0.1))
    
    return logp
    
#Set up walkers
ndim = 8
ntemps = 1
nwalkers = 32

#Set up initial point
p0PT = np.random.uniform(low=0.0, high=1.0, size=(ntemps, nwalkers, ndim))
p0PT = np.array(p0PT)
x0 = np.copy(p0PT[0,0,:])
x0[6] = 0.75
x0[7] = 0.99
x0[4] = 100.

sampler = PTSampler(ntemps, nwalkers, ndim, logl, lnprior, threads=32,  loglargs=[vo,so,vt,st,co])

# Burn-in
for p, lnprob, lnlike in sampler.sample(p0PT, iterations=1000):
    pass
sampler.reset()

# Sample
for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                       lnlike0=lnlike,
                                       iterations=10000, thin=10):
    pass
    
assert sampler.chain.shape == (ntemps, nwalkers, 1000, ndim)
pickle.dump(sampler.chain, open("equateurFullChainWithSero.p", "wb"))
