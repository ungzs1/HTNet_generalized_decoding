"""
Train and fine-tune decoders. Should work with other datasets as long as they are
in the same xarray format (will need to specify loadpath too).
"""

from run_nn_models import run_nn_models

import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use

t_start = time.time()

# Where data will be saved: rootpath + dataset + '/'
rootpath = '/media/ungzs10/F8426F05426EC7C8/Zsombi/MTA/ecog_local/data/naturalistic_move_v_rest/'
dataset = 'move_rest_ecog'

# Data load paths
ecog_lp = rootpath + 'ecog_dataset/'  # data load path
ecog_roi_proj_lp = ecog_lp+'proj_mat/' #

# patient data
pats_ids_in = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10', 'EC11', 'EC12']
tlim = [-1, 1]

# *** Tailored decoder parameters (within participant) ***

n_folds_tail = 3  # number of folds (per participant)

# testing modes
spec_meas_tail = ['power', 'power_log', 'relative_power', 'phase', 'freqslide']

# hyperparameters
hyps_tail = {'F1': 20,
             'dropoutRate': 0.693,
             'kernLength': 64,
             'kernLength_sep': 56,
             'dropoutType': 'SpatialDropout2D',
             'D': 2,
             'n_estimators': 240,
             'max_depth': 9}

hyps_tail['F2'] = hyps_tail['F1'] * hyps_tail['D']  # F2 = F1 * D
epochs_tail = 300
patience_tail = 30

# todo ???
sp_finetune = [rootpath + dataset + '/tf_all_per/',
               rootpath + dataset + '/tf_per_1dconv/',
               rootpath + dataset + '/tf_depth_per/',
               rootpath + dataset + '/tf_sep_per/']  # where to save output (should match layers_to_finetune)

# How much train/val data to use, either by number of trials or percentage of available data
use_per_vals = True  # if True, use percentage values (otherwise, use number of trials)
per_train_trials = [.17, .33, .5, 0.67]
per_val_trials = [.08, .17, .25, 0.33]
n_train_trials = [16, 34, 66, 100]
n_val_trials = [8, 16, 34, 50]

# *** Tailored decoder training ***
for s, val in enumerate(spec_meas_tail):
    do_log = True if val == 'power_log' else False
    compute_val = 'power' if val == 'power_log' else val
    single_sp = rootpath + dataset + '/single_sbjs_' + val + '/'
    combined_sbjs = False
    if not os.path.exists(single_sp):
        os.makedirs(single_sp)
    if s == 0:  # avoid fitting non-HTNet models again
        models = ['eegnet_hilb', 'eegnet', 'rf', 'riemann']
    else:
        models = ['eegnet_hilb']

    run_nn_models(sp=single_sp,  # save directory for current patient
                  n_folds=n_folds_tail,  # number of folds (per participant)
                  combined_sbjs=combined_sbjs,  # False if not train between multiple patients
                  lp=ecog_lp,  # data load path
                  roi_proj_loadpath=ecog_roi_proj_lp,  # projection matrix load path
                  pats_ids_in=pats_ids_in,
                  test_day='last',
                  tlim=tlim,
                  do_log=do_log,  # True for 'power_log' measurement
                  epochs=epochs_tail, patience=patience_tail, models=models, compute_val=compute_val,
                  F1=hyps_tail['F1'], dropoutRate=hyps_tail['dropoutRate'], kernLength=hyps_tail['kernLength'],
                  kernLength_sep=hyps_tail['kernLength_sep'], dropoutType=hyps_tail['dropoutType'],
                  D=hyps_tail['D'], F2=hyps_tail['F2'], n_estimators=hyps_tail['n_estimators'],
                  max_depth=hyps_tail['max_depth'])
'''
# Combine results into dataframes
ntrain_combine_df(rootpath + dataset)
frac_combine_df(rootpath + dataset, ecog_roi_proj_lp)
'''
'''
#### Pre-compute difference spectrograms for ECoG and EEG datasets ####
diff_specs(rootpath + dataset + '/combined_sbjs_power/', ecog_lp, ecog=True)
diff_specs(rootpath + dataset + '/combined_sbjs_power/', eeg_lp, ecog=False)
'''
print('Elapsed time: ' + str(time.time() - t_start))
