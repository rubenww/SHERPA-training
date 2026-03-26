#!/usr/bin/env python
# encoding: utf-8 
import os
import sys
import time
import numpy as np
import sherpa.read_scenarios.ReadScenarios as rs
import sherpa.read_scenarios.computeDistanceRatio as cr
import sherpa.training.step2.step2 as s2
import sherpa.validation.validation as v
from optparse import OptionParser

#20230206 
# I am now using this general function, that allows to work with
# - yearly values, low and high sources together
# - seasonal values (DJF, MAM, JJA, SON), low and high sources together
# - yearly values, low and high sources split

# chooseModel =  'wrfchem_china_27kmres_2023'
#'wrfchem_china_27kmres_2023'  # this in the case you use low and high sources summed up
# chooseModel = 'emepV434_camsV42withCond_01005_month'
# chooseModel = 'emepV4_45_cams61_withCond_01005_2021'#
# chooseModel = 'met21_emep445_cams2015_42nocond'#'emep45_edgar_2022' #'met15_emep50_cams2015v80'#'emep45_cams80'#
# chooseModel = 'emep45_cams80'#
chooseModel = 'rivm'#
               
#20230206 define if to split low and high level sources
split_low_high_sources = False
if split_low_high_sources :
    #source_split=['', '_low', '_high'] 
    source_split=['_low', '_high'] 
else :
    source_split=[''] \
    
#20230206 define if to consider only yearly, or also seasonal indicators
time_agg_period = ['yearly', 'monthly', 'monthly', 'monthly', 'monthly']
time_agg_tag = ['YEA', 'DJF', 'MAM', 'JJA', 'SON']    
start_time_loop = 0; end_time_loop = 1 #0,1 means you run only yearly values - 0,5 means YEA + 4 seasons
# start_time_loop = 0; end_time_loop = 1 #0,1 means you run only yearly values - 0,5 means YEA + 4 seasons

#20230206 list of SRR to be tested
# aqi_to_be_tested = list([0,1,2,5,6])
# aqi_to_be_tested = list([5])

#20230206 standard optimization to be performed
chooseOpt = 'step1_omegaPerPoll_aggRes_perPoll'        

#20230206 only emepV434_camsV42withCond_01005_month is currently used
if chooseModel == 'rivm':
    import sherpa.configuration_emep4nl_2025 as c
elif chooseModel == 'emep10km':
    import sherpa.configuration_emep as c
elif chooseModel == 'emepV434_camsV42withCond_01005_month':
    import sherpa.configuration_emepV434_camsV42withCond_01005_month as c
elif chooseModel == 'emepV4_45_cams61_withCond_01005_2019':
    import sherpa.configuration_emepV4_45_cams61_withCond_01005_2019 as c
elif chooseModel == 'emepV4_45_cams61_withCond_01005_2019_meteo2021':
    import sherpa.configuration_emepV4_45_cams61_withCond_01005 as c
elif chooseModel == 'emepV4_45_cams61_withCond_01005_2019_meteo2019':
    import sherpa.configuration_emepV4_45_cams61_withCond_01005 as c
elif chooseModel == 'emepV4_45_cams61_withCond_01005_2019_meteo2017':
    import sherpa.configuration_emepV4_45_cams61_withCond_01005 as c    
elif chooseModel == 'emepV4_45_cams61_withCond_01005_2019_meteo2015':
    import sherpa.configuration_emepV4_45_cams61_withCond_01005 as c    
elif chooseModel == 'wrfchem_china_27kmres_2023':
    import sherpa.configuration_wrfchem_china_27kmres_2023 as c
elif chooseModel == 'emep45_cams80':
    import sherpa.configuration_EMEP_45_CAMSv80_01005 as c    
elif chooseModel == 'emep45_cams80_emi2015':
    import sherpa.configuration_EMEP_45_CAMSv80_01005_emi2015 as c    
# elif chooseModel == 'met15_emep50_cams2015v80':
#     import sherpa.config_met15_emep50_cams2015v80 as c    
# elif chooseModel == 'met15_emep434_cams2015v80':
#     import sherpa.config_met15_emep434_cams2015v80 as c    
# elif chooseModel == 'emep45_edgar_2022':
#     import sherpa.config_emep45_edgar_2022 as c    
# elif chooseModel == 'emep45_emep_2022':
#     import sherpa.config_emep45_emep_2022 as c    
# elif chooseModel == 'met21_emep445_cams2015_42nocond':
#     import sherpa.config_met21_emep445_cams2015_42nocond as c    
   
#20230206 only 'step1_omegaPerPoll_aggRes_perPoll' is currently used    
if chooseOpt == 'step1_omegaPerPoll_aggRes':
    import sherpa.training.step1.step1_omegaPerPoll_aggRes as s1
elif chooseOpt == 'step1_omegaPerPoll_aggRes_perPoll':
    import sherpa.training.step1.step1_omegaPerPoll_aggRes_perPoll as s1
elif chooseOpt == 'step1_omegaPerPoll_aggRes_perPoll_ch':
    import sherpa.training.step1.step1_omegaPerPoll_aggRes_perPoll_CH as s1

__all__ = []
__version__ = 0.1
__date__ = '2017-01-13'
__updated__ = '2017-01-25'

def main(argv=None):
    # Command line options.
    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = "%s" % __updated__
    program_version_string = '%%prog %s (%s)' % (program_version, program_build_date)
    program_longdesc = '''''' # optional - give further explanation about what the program does
    program_license = "Copyright 2017 ISPRA                                                    \
            Licensed under the Apache License 2.0 \
            http://www.apache.org/licenses/LICENSE-2.0"

    if argv is None:
        argv = sys.argv[1:]

    # setup option parser
    parser = OptionParser(version=program_version_string, epilog=program_longdesc, description=program_license)
    parser.add_option("-p", "--path", dest="datapath", help="set data path [default: %default]", metavar="DIR")
    parser.add_option("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %default]")
    parser.add_option("-m", "--mode", dest="mode", help="set mode (T:training and validation, V:validation) [default: %default]")
    parser.add_option("-i", "--indicator", dest="indicator", help="set indicators to be trained", action='append', type='int')
    parser.add_option("-o", "--omega", dest="omega", help="initial omega", type='float')
    parser.add_option("-r", "--run", dest="run", help="run name", type='string')

    # for defaults
    conf = c.Config()
    conf = c.configuration(conf, chooseModel, chooseOpt, time_agg_period[0], time_agg_tag[0], 0, source_split[0]);

    try:    
        parser.set_defaults(datapath=conf.datapath, mode=conf.mode)
        (opts, args) = parser.parse_args(argv)
    except Exception as e:
        indent = len(program_name) * " "
        # print(traceback.format_exc())
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        #print(sys.exc_info()[0])
        return 2

    #loop on air quality indicators
    aqi_to_be_tested = opts.indicator

    ############################

    if opts.verbose and opts.verbose > 0:
        print("verbosity level = %d" % opts.verbose, flush=True)
    if opts.datapath:
        print("datapath = %s" % opts.datapath, flush=True)
    if opts.mode:
        print("mode = %s" % opts.mode, flush=True)

    start = time.time();

    os.chdir(opts.datapath)
    
    #20230206 train 2 SRR in case of low and high level split, only 1 SRR of no split
    for source_split_instance in source_split:
        
        #20230206 loop to compute only YEA, or also seasonal (DJF, MAM, JJA, SON) indicators
        for aqi_selected in aqi_to_be_tested:
            
            #loop on time aggregation
            for iter_loop in range(start_time_loop, end_time_loop):
                print('processing ' + str(time_agg_period[iter_loop]), ', indicator ' + str(aqi_selected), flush=True)

                conf = c.Config()
                conf.mode = opts.mode;
                conf.datapath = opts.datapath
                conf.omega_guess = opts.omega
                conf.run = opts.run
                conf = c.configuration(conf, chooseModel, chooseOpt, time_agg_period[iter_loop], time_agg_tag[iter_loop], aqi_selected, source_split_instance)
            
                print('read_scenarios', flush=True);
                rs.ReadScenarios(conf);
        
                #compute ratio useful to correct weighting factor matrix, in the case of lat lon
                cr.computeDistanceRatio(conf)
        
                if opts.mode=='T':
                    print('step1', flush=True);
        
                    #this uses varying omega
                    s1.step1_omegaOptimization(conf)
        
        #           # this is the test done during December 2019, using fixed omega
                    # conf.omegaFinalStep1_alldom = np.zeros((conf.Prec.shape[0], conf.Prec.shape[1], 5));
                    # conf.omegaFinalStep1 = np.zeros_like(conf.omegaFinalStep1_alldom)
                    # conf.omegaFinalStep1[:] = 2.5 #if you want to consider 1.5 fix
        
                    print('step2', flush=True);
                    s2.step2(conf);
        
                print('validation', flush=True);
                v.validation(conf);
                ############################
        
    print('end');
    print("Execution time: %s" % (time.time() - start));
            

    

if __name__ == "__main__":
    sys.exit(main())
