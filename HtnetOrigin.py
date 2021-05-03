from abc import ABC

from HtnetBase import HtnetBase


class HtnetOriginal(HtnetBase):

    def __init__(self, *args, **kwargs):
        super(HtnetBase, self).__init__(*args, **kwargs)

    def set_metadata(self):
        """
        set database specific metadata here.
        """

        # Where data will be saved: rootpath + dataset + '/'
        self.rootpath = '/media/ungzs10/F8426F05426EC7C8/Zsombi/MTA/ecog_local/data/naturalistic_move_v_rest/'
        self.dataset = 'move_rest_ecog'

        # Data load paths
        self.lp = self.rootpath + 'ecog_dataset/'  # data load path
        self.ecog_roi_proj_lp = self.lp + 'proj_mat/'  #

        # patient data
        self.pats_ids_in = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10', 'EC11',
                            'EC12']
        self.tlim = [-1, 1]
        self.test_day = 'last'

        # *** Tailored decoder parameters (within participant) ***
        self.hyps_tail = {'F1': 20,
                          'dropoutRate': 0.693,
                          'kernLength': 64,
                          'kernLength_sep': 56,
                          'dropoutType': 'SpatialDropout2D',
                          'D': 2,
                          'n_estimators': 240,
                          'max_depth': 9}
        self.hyps_tail['F2'] = self.hyps_tail['F1'] * self.hyps_tail['D']  # F2 = F1 * D
        self.epochs_tail = 30
        self.patience_tail = 30


if __name__ == '__main__':
    HtnetOriginal.train_decoders()
