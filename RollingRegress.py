#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:57:28 2017

@author: Mich
"""


import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pickle

#full_return.to_pickle('full_return')
full_return = pd.read_pickle('full_return')

sp1 = full_return.iloc[:6900,1]
sp2 = full_return.iloc[:6900,2]

model_randomwalk = pm.Model()
with model_randomwalk:
    # std of random walk, best sampled in log space.
    sigma_alpha = pm.Exponential('sigma_alpha', 1./.02, testval = .1)
    sigma_beta = pm.Exponential('sigma_beta', 1./.02, testval = .1)
    
import theano.tensor as T

# To make the model simpler, we will apply the same coefficient for 50 data points at a time
subsample_alpha = 50
subsample_beta = 50

with model_randomwalk:
    alpha = pm.GaussianRandomWalk('alpha', sigma_alpha**-2,
                                  shape=len(sp1) // subsample_alpha)
    beta = pm.GaussianRandomWalk('beta', sigma_beta**-2,
                                 shape=len(sp1) // subsample_beta)

    # Make coefficients have the same length as prices
    alpha_r = T.repeat(alpha, subsample_alpha)
    beta_r = T.repeat(beta, subsample_beta)
    
with model_randomwalk:
    # Define regression
    regression = alpha_r + beta_r * sp1

    # Assume prices are Normally distributed, the mean comes from the regression.
    sd = pm.Uniform('sd', 0, 20)
    likelihood = pm.Normal('y',
                           mu=regression,
                           sd=sd,
                           observed=sp2) 
from scipy import optimize
    
with model_randomwalk:
    # First optimize random walk
    start = pm.find_MAP(vars=[alpha, beta], fmin=optimize.fmin_l_bfgs_b)

    # Sample
    step = pm.NUTS(scaling=start)
    trace_rw = pm.sample(2000, step, start=start)
    
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111, xlabel='time', ylabel='alpha', title='Change of alpha over time.')
ax.plot(trace_rw[-1000:][alpha].T, 'r', alpha=.05);
ax.set_xticklabels([str(p.date()) for p in prices[::len(prices)//5].index]);

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlabel='time', ylabel='beta', title='Change of beta over time')
ax.plot(trace_rw[-1000:][beta].T, 'b', alpha=.05);
ax.set_xticklabels([str(p.date()) for p in prices[::len(prices)//5].index]);

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlabel='Price GDX in \$', ylabel='Price GLD in \$',
            title='Posterior predictive regression lines')

colors = np.linspace(0.1, 1, len(prices))
colors_sc = np.linspace(0.1, 1, len(trace_rw[-500::10]['alpha'].T))
mymap = plt.get_cmap('winter')
mymap_sc = plt.get_cmap('winter')

xi = np.linspace(prices.GDX.min(), prices.GDX.max(), 50)
for i, (alpha, beta) in enumerate(zip(trace_rw[-500::10]['alpha'].T, trace_rw[-500::10]['beta'].T)):
    for a, b in zip(alpha, beta):
        ax.plot(xi, a + b*xi, alpha=.05, lw=1, c=mymap_sc(colors_sc[i]))

sc = ax.scatter(prices.GDX, prices.GLD, label='data', cmap=mymap, c=colors)
cb = plt.colorbar(sc)
cb.ax.set_yticklabels([str(p.date()) for p in prices[::len(prices)//10].index]);