from methods.utils.SELD_metrics import *
from utils.ddp_init import reduce_value

class Metrics(object):
    """Metrics for evaluation

    """
    def __init__(self, dataset):

        # self.metrics = []
        self.names = ['ER_macro', 'F_macro', 'LE_macro', 'LR_macro', 'SELD_scr_macro', 'ER_micro', 'F_micro', 'LE_micro', 'LR_micro', 'SELD_scr_micro']

        self.num_classes = dataset.num_classes
        self.doa_threshold = 20 # in deg
        self.num_frames_1s = int(1 / dataset.label_resolution)
        self.metrics = SELDMetrics(nb_classes=self.num_classes, doa_threshold=self.doa_threshold)

    def update(self, pred_dict, gt_dict):
        self.metrics.update_seld_scores(pred_dict, gt_dict)
        
    def calculate(self):

        # ER: error rate, F: F1-score, LE: Location error, LR: Location recall
        self.metrics._average = 'macro'
        ER_macro, F_macro, LE_macro, LR_macro, seld_score_macro, _ = self.metrics.compute_seld_scores()
        self.metrics._average = 'micro'
        ER_micro, F_micro, LE_micro, LR_micro, seld_score_micro, _ = self.metrics.compute_seld_scores()
        self.metrics = SELDMetrics(nb_classes=self.num_classes, doa_threshold=self.doa_threshold)

        metrics_scores_macro = {
            'ER_macro': ER_macro,
            'F_macro': F_macro,
            'LE_macro': LE_macro,
            'LR_macro': LR_macro,
            'seld_macro': seld_score_macro,
        }
        metrics_scores_micro = {
            'ER_micro': ER_micro,
            'F_micro': F_micro,
            'LE_micro': LE_micro,
            'LR_micro': LR_micro,
            'seld_micro': seld_score_micro,
        }
        metrics_scores = {
            'macro': metrics_scores_macro,
            'micro': metrics_scores_micro,
        }
        return metrics_scores
