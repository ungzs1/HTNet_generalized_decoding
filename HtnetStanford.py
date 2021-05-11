import numpy as np

from HtnetBase import *

from model_utils import load_data


class HtnetStanford(HtnetBase):
    def __init__(self, *args, **kwargs):
        super(HtnetStanford, self).__init__()

        my_path = os.path.abspath(os.path.dirname(__file__))

        # Where data will be saved
        dataset = 'stanford_3trial'
        self.save_rootpath = os.path.join(my_path, '..', 'ecog', 'trained_models', 'HTNet', dataset)

        # Data load paths
        load_rootpath = os.path.join(my_path, '..', 'ecog', 'data', 'raw_data', 'stanford_motorbasic')
        self.lp = os.path.join(load_rootpath)

        # patient ids
        self.pats_ids_in = ['bp', 'ca', 'cc', 'de', 'fp', 'gc', 'hh', 'hl', 'jc', 'jm', 'jt', 'rh', 'rr', 'ug', 'wc']

        # model settings
        self.models = ['eegnet_hilb', 'eegnet']
        self.spec_meas_tail = ['power']#, 'power_log', 'phase', 'freqslide']  #, 'relative_power' todo relative powerre nem mukodik a htnet_model

    def set_tailored_hyperparameters(self):
        """ Tailored decoder parameters (within participant) """
        self.n_folds = 3
        self.hyps = {'F1': 20,
                     'dropoutRate': 0.693,
                     'kernLength': 64,
                     'kernLength_sep': 56,
                     'dropoutType': 'SpatialDropout2D',
                     'D': 2,
                     'n_estimators': 240,
                     'max_depth': 9}
        self.hyps['F2'] = self.hyps['F1'] * self.hyps['D']  # F2 = F1 * D
        self.epochs = 30
        self.patience = 30

    def load_data(self, patient, randomize_events=True):
        """
        Database specific load method for given patient. Load data from file.
        :param patient: current patient id.
        :param randomize_events: set 'True' to randomize events, set 'False' to keep event order. Default is 'True'.
        :return: X: train data in the form of (trial, channel, timestep)
        :return: y: train labels in the form of (trial, )
        :return: X_test: test data in the form of (trial, channel, timestep)
        :return: y_test: test labels in the form of (trial, )
        """
        
        import sys, os
        import pathlib
        abspath = pathlib.Path(__file__).parent.absolute()
        sys.path.insert(0, os.path.join(abspath, '..', 'ecog', 'preprocessing'))

        from preprocessor_stanford_motor import StanfordPreprocessor

        sp = StanfordPreprocessor()
        sp.test_size = 0.2  # set test size, default is test_size=0.2
        X, y, X_test, y_test = sp.get_train_test_data(patient)

        # Randomize event order (random seed facilitates consistency)
        if randomize_events:
            order_inds = np.arange(len(y))
            np.random.shuffle(order_inds)
            X = X[order_inds, ...]
            y = y[order_inds]

            order_inds_test = np.arange(len(y_test))
            np.random.shuffle(order_inds_test)
            X_test = X_test[order_inds_test, ...]
            y_test = y_test[order_inds_test]

        # transform labels to 0-3 range
        for i, label in enumerate(y):
            if label == 11: y[i] = 0
            if label == 12: y[i] = 1
            if label == 110: y[i] = 2
            if label == 120: y[i] = 3

        for i, label in enumerate(y_test):
            if label == 11: y_test[i] = 0
            if label == 12: y_test[i] = 1
            if label == 110: y_test[i] = 2
            if label == 120: y_test[i] = 3

        # remove specific labels
        label_to_remove = 3
        X = np.asarray([data for i, data in enumerate(X) if not y[i]==label_to_remove])
        y = np.asarray([label for label in y if not label==label_to_remove])
        X_test = np.asarray([data for i, data in enumerate(X_test) if not y_test[i]==label_to_remove])
        y_test = np.asarray([label for label in y_test if not label==label_to_remove])

        return X, y, X_test, y_test


if __name__ == '__main__':
    HtnetStanford().train_decoders()
