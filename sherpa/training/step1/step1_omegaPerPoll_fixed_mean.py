'''
Created on 6-feb-2017
Modified the 20170321, by EP

@author: roncolato
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

def InvDistN_opt_prec(beta,xdata,rad,latVecFilt, poly):
    
    ratio = np.polyval(poly, latVecFilt)
    Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
    F = 1 / ((1 + ((X / ratio) ** 2 + Y ** 2) ** 0.5));
    F = F**beta[1]
    output = beta[0] * np.inner(xdata, F.flatten());
    
    return output;

def iop(beta,inp1,inp2,rad, latVecFilt, poly):
    x=InvDistN_opt_prec(beta,inp1,rad,latVecFilt, poly)
    y=inp2.flatten().T
#    print(x)
#    print(y)
#    print(sqrt(mean_squared_error(y, x)))
    return sqrt(mean_squared_error(y, x))
#    return np.mean(((x - y.T) ** 2))


def step1_omegaOptimization(conf):
    res_step = 1
    #convert from 28 to 7 km
    Prec = f7.from7to28(conf.Prec, res_step);
    ny = int(round(conf.ny/res_step))
    nx = int(round(conf.nx/res_step))
    rad = conf.radStep1;
    nPrec = conf.nPrec
    rf = conf.rf1
    flagRegioMat = np.copy(conf.flagRegioMat);

    #pad Prec with zeros around initial matrix, to perform matrix products later on
    Prec2 = np.zeros((ny+rad*2,nx+rad*2,Prec.shape[2],Prec.shape[3]));
    Prec2[rad:-rad,rad:-rad,:,:] = Prec[:,:,:,:];
    Prec=Prec2;

    print("res_step={}; nx={}; ny={}; rf={}; rad={}".format(res_step, nx, ny, rf, rad), flush=True)

    #convert from 28 to 7 km
    Indic = f7.from7to28(conf.Indic, res_step);
    flagRegioMat = f7.from7to28(flagRegioMat, res_step);
    lat = f7.from7to28(conf.y, res_step);
    # flagPerNoxPP??m = f7.from7to28(flagPerNoxPPm, res_step);

    #initialize variables
    omega = np.full([ny,nx,nPrec],np.nan);
#    alphaTmp = np.zeros((categories.size));
#    omegaTmp = np.zeros((categories.size));

    #define training scenarios; note scenarios number is +1 if checking DoE...as in line 74 it is -1
    if conf.domain == 'emep4nl_2025':
        # conf.Order_Pollutant = '0=NOx, 1=NMVOC, 2=NH3, 3=PPM2.5, 4=PPMco, 5=SOx'
        # TODO: NMVOC
        # = run number - 1
        IdeVec = (np.array([1, 3]),np.array([1, 3]),np.array([1, 2]),np.array([1, 5]),np.array([1, 6]),np.array([1, 4]));
    elif conf.domain == 'emep10km':
        if conf.aqi == 'SURF_ug_PM25_rh50-Yea':
            IdeVec = (np.array([1, 1]),np.array([1, 2]),np.array([1, 3]),np.array([1, 5]),np.array([1, 6]));
        elif conf.aqi == 'SURF_ug_PM10_rh50-Yea':
            IdeVec = (np.array([1, 1]),np.array([1, 2]),np.array([1, 3]),np.array([1, 4]),np.array([1, 6]));
    elif conf.domain == 'ineris7km':
        IdeVec = (np.array([1, 2]),np.array([1, 3]),np.array([1, 4]),np.array([1, 5]),np.array([1, 6]));
    # elif (conf.domain == 'emepV433_camsV221')  | (conf.domain == 'edgar2015') | (conf.domain == 'emepV434_camsV42'):
    elif ('emep' in conf.domain) |  (conf.domain == 'edgar2015') | ('wrf' in conf.domain):        
        IdeVec = (np.array([1, 1]), np.array([1, 2]), np.array([1, 3]), np.array([1, 4]), np.array([1, 5]));

    #loop over precursors
    print('precursors: {}'.format(conf.PrecToBeUsed), flush=True)
    print('omega: {}'.format(conf.omega_guess), flush=True)
    if len(conf.omega_guess) != len(conf.PrecToBeUsed):
        print('wrong number of initial omegas!', flush=True)
        exit()

    for pi, precursor in enumerate(conf.PrecToBeUsed):
        
        PREC = precursor;
        Ide = IdeVec[precursor];
        nSc = Ide.shape[0]-1;# size(Ide,2)-1
#        icel = 0;
        
        #20220414, test with decreased bounds
        # bnds = ((0, 1), (1.5, 2.5)) #20220524, used for PM25, PM10, O3
        # bnds = ((0, 1), (0.5, 2.5)) #20220524, used for PM25, PM10, O3
        # bnds = ((0, 1), (1.75, 2.5)) #20220524, used for NO2 and NO
        # bnds = ((0, 1), (1.5, 3)) #20220524, used for NO2 and NO
        
        #VERSION USED FOR ALL TESTS IN 2025 
        bnds = ((None, None), (conf.omega_guess[pi]-0.1, conf.omega_guess[pi]+0.1)) # RV: default so far
        #VERSION USED FOR ALL TESTS IN 2025 

        print('precursor: {} {:.2f}'.format(PREC, conf.omega_guess[pi]), flush=True)

        for ic in range(0, nx):
            for ir in range(0, ny):
                if flagRegioMat[ir,ic]>0:
                    #create data for omega calculation
                    tmpPrec = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide[1],PREC],rad); # patches
                    tmpInde = ei.EquaIndic(ic,ir,rf,nx,ny,nSc,Indic[:,:,Ide[1]]); # indicator
                    mdl = minimize(iop, [1, conf.omega_guess[pi]], args=(tmpPrec, tmpInde, rad, lat[ir,ic], conf.ratioPoly), bounds=bnds, method='SLSQP', options={'disp': False})  # L-BFGS-B, TNC
                    omega[ir,ic,PREC] = mdl.x[1]

            print("{}/{} {}/{}: omega: {:.2f}".format(PREC+1, conf.nPrec, ic+1, nx, np.nanmean(omega[:,ic,PREC])), flush=True);
        
    #rescale to initial spatial resolution, through nearest interpolation
    #initialize variable
    omegaFinal = np.zeros((conf.Prec.shape[0], conf.Prec.shape[1], conf.nPrec));

    # only save mean omega
    for poll in range(0, nPrec):
        mean_omega = np.nanmean(omega[:,:,poll])
        print("Poll {} ; omega = {}".format(poll, round(mean_omega, 2)))
        omegaFinal[:,:,poll] = mean_omega
    
    #keep only results on the mask
    omegaFinal[conf.flagRegioMat==0] =np.nan    
    
    conf.omegaFinalStep1 = omegaFinal
    conf.omegaFinalStep1_notFiltered = omegaFinal
    conf.ci2Step1 = [];
    conf.CovB2Step1 = [];
