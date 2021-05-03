import os
import time
from htnet_model import htnet
from model_utils import load_data


class HtnetBase:
    def __init__(self):
        # database information
        self.rootpath = None
        self.dataset = None
        self.lp = None
        self.ecog_roi_proj_lp = None
        self.pats_ids_in = []
        self.tlim = None
        self.test_day = None

        # tailored decoder parameters
        self.combined_sbjs = False
        self.n_folds_tail = 3
        self.hyps_tail = {'F1': 20,
                          'dropoutRate': 0.693,
                          'kernLength': 64,
                          'kernLength_sep': 56,
                          'dropoutType': 'SpatialDropout2D',
                          'D': 2,
                          'n_estimators': 240,
                          'max_depth': 9,
                          'F2': None}
        self.epochs_tail = 30
        self.patience_tail = 30
        self.spec_meas_tail = ['power', 'power_log', 'relative_power', 'phase', 'freqslide']

        # decoder training parameters
        self.sp = ''
        self.do_log = ''
        self.compute_val = ''

        # ???
        self.n_evs_per_sbj = ''
        self.n_chans_all = ''
        self.dipole_dens_thresh = ''
        self.rem_bad_chans = ''
        self.models = ''
        self.save_suffix = ''
        self.overwrite = ''
        self.F2 = ''
        self.rand_seed = ''
        self.loss = ''
        self.optimizer = ''
        self.patience = ''
        self.early_stop_monitor = ''
        self.n_test = ''
        self.n_val = ''
        self.n_train = ''
        self.epochs = ''
        self.ecog_srate = ''
        self.trim_n_chans = ''


    def set_metadata(self):
        raise NotImplementedError

    def cnn_model(self):
        raise NotImplementedError

    def run_nn_models(self):
        raise NotImplementedError

    def train_decoders(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use

        #t_start = time.time()

        # *** Tailored decoder training ***
        for s, val in enumerate(self.spec_meas_tail):
            self.do_log = True if val == 'power_log' else False
            self.compute_val = 'power' if val == 'power_log' else val
            self.sp = self.rootpath + self.dataset + '/single_sbjs_' + val + '/'

            if not os.path.exists(self.sp):
                os.makedirs(self.sp)
            if s == 0:  # avoid fitting non-HTNet models again
                self.models = ['eegnet_hilb', 'eegnet', 'rf', 'riemann']
            else:
                self.models = ['eegnet_hilb']

            self.run_nn_models()

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



