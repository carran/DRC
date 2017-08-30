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
#%matplotlib inline

#Real data, read in with Pandas.
vobs_pandas = pd.read_csv('Xvax.csv')
vtrials_pandas = pd.read_csv('Nvax.csv')
sobs_pandas = pd.read_csv('Ysero.csv')
strials_pandas = pd.read_csv('Nsero.csv')
cases_pandas = pd.read_csv('reported_cases_all.csv')


#Select the locations you want to work with
loc_subset = ['ORIENTALE']
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

ndim = 8

#Load the chains
chains = pickle.load(open("chains/orientaleFullChainWithSero.p", "rb"))
samples = chains[0,0,:,:]

x = np.linspace(1,60,60)
cplot = np.zeros(60)
vplot = np.zeros(60)
splot = np.zeros(60)
ctest = np.zeros(60)
l = len(samples[:,0])
nplot = 500
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,10))
#ax0 = axes.flatten()
ax0, ax1, ax2 = axes.flatten()
#ax0, ax1 = axes.flatten()
for k in range(nplot):
    index = np.random.randint(l)
    
    valpha = samples[index,0]
    vbeta = samples[index,1]
    vheight = samples[index,2]
    falpha = samples[index,3]
    fbeta = samples[index,4]
    fheight = samples[index,5]
    #fheight = 25.
    veff = samples[index,6]
    seff = samples[index,7]
    falpha2 = 0.31
    fbeta2 = 52.
    fheight2 = 11.
    
    j = 0
    for i in x:
        vplot[j] = (1. - np.exp(-vheight*(1. - np.exp(-((float(i))/vbeta)**valpha)))) # * veff
        splot[j] = 1.0 - np.exp(-vheight*veff*(1. - np.exp(-((float(i))/vbeta)**valpha)) - 
                                          fheight*(1. - np.exp(-(float(i))/fbeta)**falpha))
        cplot[j] = np.exp(-(fheight*(falpha/fbeta) * (float(i+1)*12./fbeta)**(falpha-1.)*np.exp(-(float(i+1)*12./fbeta)**falpha)))*\
        (np.exp(-veff*vheight*(1.-np.exp(-(float(i+1)*12./vbeta)**valpha))-fheight*(1.-np.exp(-(float(i+1)*12./fbeta)**falpha))))
        #fheight = 0.
        ctest[j] = np.exp(-(fheight2*(falpha2/fbeta2) * (float(i+1)*12./fbeta2)**(falpha2-1.)*np.exp(-(float(i+1)*12./fbeta2)**falpha2)))*\
        (np.exp(-veff*vheight*(1.-np.exp(-(float(i+1)*12./vbeta)**valpha))-fheight2*(1.-np.exp(-(float(i+1)*12./fbeta2)**falpha2))))
        j += 1

    ax0.plot(x,vplot,alpha=0.02,color='b')
    ax1.plot(x,splot,alpha=0.01,color='g',linewidth=1)
    ax2.plot(cplot/sum(cplot),alpha=0.01,color='c',linewidth=1)
    #ax2.plot(ctest/sum(ctest),alpha=0.02,color='r')
    #ax1.plot(softc,alpha=0.01,color='c')
        
ax0.plot(vo.astype(float)/vt.astype(float), 'ro')
ax1.plot(so.astype(float)/st.astype(float),'ro')
ax2.plot(co.astype(float)/sum(co),'ro')

fig.tight_layout()
plt.savefig('figures/orientaleVaxCasesAgeSero.png')
