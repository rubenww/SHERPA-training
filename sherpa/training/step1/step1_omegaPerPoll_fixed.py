'''
Fixed omega per pollutant
'''

import numpy as np
import scipy.interpolate as interpol
from scipy.optimize import minimize

from sherpa.training.step1 import from7to28 as f7
from sherpa.training.step1 import EquaPrec as ep
from sherpa.training import EquaIndic as ei
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.ndimage as gf


def step1_omegaOptimization(conf):
    nPrec = conf.nPrec
    ny = conf.ny
    nx = conf.nx

    #loop over precursors
    print('precursors: {}'.format(conf.PrecToBeUsed), flush=True)
    print('omega: {}'.format(conf.omega_guess), flush=True)
    if len(conf.omega_guess) != len(conf.PrecToBeUsed):
        print('wrong number of initial omegas!', flush=True)
        exit()

    #initialize variables
    omega = np.full([ny,nx,nPrec],np.nan) conf.omega_guess);
    for i, pi in enumerate(conf.PrecToBeUsed):
        omega[:,:,pi] = conf.omega_guess[i]
    
    conf.omegaFinalStep1_notFiltered = omega
    conf.omegaFinalStep1 = omega
    conf.ci2Step1 = []
    conf.CovB2Step1 = []
