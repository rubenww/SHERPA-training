'''
Created on 13-mar-2017
define configuration of the training and validation run
Edited by Ruben Verweij (RIVM)
@author: roncolato
'''
import numpy as np
import platform
from datetime import datetime

dirprefix = ""

#class configuration defines methods and attributes
#methods: used to create names of scenarios to be loaded
class Config:
    def __init__(self):
        # overwritable defaults
        self.root = '{}/EMEP/users/verweijr/_EMEP_projects/20250117_schaduw_GCN_EMEP/20260619_PPM2.5_tests/level04/training/'.format(dirprefix)
        self.datapath = self.root
        self.run = 'run001'
        self.mode = 'T'

    def scenEmissionFileName(self, sce):
        sces = '%03i'%(sce);
        root = '{}/EMEP/EMEP/rv4.45/run/schaduw_gcn/for_SHERPA/set01/sce'.format(dirprefix)+sces+'.nc'
        return root
    def scenConcFileName(self, sce):
        return self.scenEmissionFileName(sce)
    pass;

def configuration(conf, chooseModel, chooseOpt, time_resol, time_loop, aqi_selected, source_split_instance):
    #ep 20200610
    conf.yearmonth = 0 #0=year, 1=month
    conf.whichmonth = '' #DJF, MAM, JJA, SON
    
    ###########################################################################
    #modify for testing
    conf.domain = 'emep4nl_2025';
    conf.flagReg = 'emep4nl_2025';
    conf.distance = 0 # 0=cells, 1=distance in km
    conf.gf = 0
    conf.rf1 = 0 # window of cells of training varying F (1=one ring of cells used for training, surrounding the target cell0
    conf.rf2 = 0
    conf.res_step = 2 # also set in step 1: resolution coarsening step factor (corresponds to ~level 03 resolution)
    ndomain = 900
    conf.radStep1 = int(round(ndomain/conf.res_step)) # number of cells to be considered in step1
    conf.radStep2 = ndomain # number of cells to be considered in step2
    conf.vec1 = [
            'SURF_ug_NH3', # 0
            'SURF_ug_NH4_F', # 1
            'DDEP_RDN_m2Grid', # 2
            'WDEP_RDN', # 3
            'SURF_ug_SO2', # 4
            'SURF_ug_SO4', # 5
            'DDEP_SOX_m2Grid', # 6
            'WDEP_SOX', # 7
            'SURF_ug_PPM_C', # 8
            'SURF_ug_PPM25', # 9
            'SURF_ug_ECCOARSE', # 10
            'SURF_ug_ECFINE', # 11
            'SURF_ug_PMCOARSE', # 12
            'SURF_ug_PM25', # 13
            'SURF_ug_NO', # 14
            'SURF_ug_NO2', # 15
            'SURF_ug_NOx', # 16
            'SURF_ug_NO3_F', # 17
            'SURF_ug_NO3_C', # 18
            'SURF_ug_HNO3', # 19
            'DDEP_OXN_m2Grid', # 20
            'WDEP_OXN', # 21
            'DDEP_NH3_m2Grid', # 22
            'DDEP_NH4_f_m2Grid', # 23
                 ]
    conf.vec3 = [
            [0,2,5], # 0
            [0,2,5], # 1
            [0,2,5], # 2
            [0,2,5], # 3
            [0,2,5], # 4
            [0,2,5], # 5
            [0,2,5], # 6
            [0,2,5], # 7
            [4], # 8
            [3], # 9
            [4], # 10
            [3], # 11
            [0,2,4,5], # 12
            [0,2,3,5], # 13
            [0,2,5], # 14
            [0,2,5], # 15
            [0,2,5], # 16
            [0,2,5], # 17
            [0,2,5], # 18
            [0,2,5], # 19
            [0,2,5], # 20
            [0,2,5], # 21
            [0,2,5], # 22
            [0,2,5] # 23
            ]

    conf.vec2 = conf.vec1
    # vec1: the output field to optimize alpha/omega for
    # vec3: which pollutants to take into account
    conf.POLLSEL = aqi_selected
    # set in other files, just for info in nc file and as a reference here
    aqiFil = conf.vec1[conf.POLLSEL]
                                         
    #NB: in case of 5=SURF_ug_NOx, NO and NO2 are summed up to produce NOx
    conf.Order_Pollutant = 'NOx, NMVOC, NH3, PPM2.5, PPMco, SOx'
    conf.nPrec = 6; # nox,sox,voc,nh3,pm25,pmco

    conf.Ide = np.arange(0,7) #training scenarios: sce(i), excl. last one: i.e. arange(0,3)=[sce000, sce001, sce002]
    conf.Val = np.arange(1,7) #validation scenarios
    conf.nSc = len(conf.Val)+1 #plus the basis run
    conf.flagRegioMatFile = conf.root+'/../flagRegioMat_NL_2025.nc'#all but #ATL	32	Remaining North-East Atlantic Ocean

    ###
    conf.nametest = chooseOpt + '_rad' + str(conf.radStep1) + '-' + str(conf.radStep2) + '_rf_' + str(conf.rf1) + '-' + str(conf.rf2) + '-distCelKm-' + str(conf.distance) + 'emiTonKm2'

    # conf.stepOptPerGroupCells_INI = 25
    # conf.stepOptPerGroupCells_REF = 5
    conf.filter = 1 #0 means do not filter omega results REF, 1 means apply gaussian filter
    ###
    if chooseOpt == 'step1_omegaPerPoll_aggRes_perPoll':
        conf.explain_step_1 = 'omega sliding per pollutant, original resolution'
        conf.explain_step_2 = 'alpha optimized per cell, all scenarios together, original resolution'
    elif chooseOpt == "no_step1_const_omega":
        conf.explain_step_1 = 'omega constant for all pollutants (no fit in step 1), original resolution'
        conf.explain_step_2 = 'alpha optimized per cell, all scenarios together, original resolution'
    elif chooseOpt == "step1_omegaPerPoll_aggRes_perPoll_const":
        conf.explain_step_1 = 'omega spatially constant fit per pollutant, all scenarios together, original resolution'
        conf.explain_step_2 = 'alpha optimized per cell, all scenarios together, original resolution'
    else:
        conf.explain_step_1 = 'unknown'
        conf.explain_step_2 = 'unknown'

    conf.alpha_physical_intepretation = 'alpha specifies the precursor relative importance';
    conf.omega_physical_intepretation = 'omega specifies the slope of the bell-shape';

    ###########################################################################

    conf.modelVariability = 1; # 1=a different model for each cell
    conf.typeOfModel = 2; # 2=regression
    conf.pcaFlag = 0; # 0 means no PCA - no Norm
    conf.absDel = 1; # absolute(0) or delta(1) values
    conf.arealPoint = 0; # 0 means areal and point summed up
    conf.flat = False; # do not use flat weight
    conf.vw = 30; #not used anymore
    conf.emiDenAbs = 0; # 0=emission density, 1=emission in absolute values

    conf.nameDirOut = conf.datapath+'/{}_output/'.format(datetime.today().strftime('%Y-%m-%d'))+conf.domain+'/'+aqiFil+'/'+conf.run+'/'+conf.nametest+'/'
    conf.nameRegFile = conf.nameDirOut+'regression.mat';

    return conf;
