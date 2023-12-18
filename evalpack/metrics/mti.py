import copy
import numpy as np
from sklearn import metrics
from scipy import special


class MTI:
    def __init__(
        self,
        recall_weight=1,
        false_alarm_weight=1,
        alarm_weight=1,
        anticipation_weight=1,
        anticipation_period="default",
        early_period="default",
        inertia_delay=100,
        recall_measure=metrics.recall_score,
        density_factor=100,
    ):
        self.recall_weight = recall_weight
        self.false_alarm_weight = false_alarm_weight
        self.alarm_weight = alarm_weight
        self.anticipation_weight = anticipation_weight
        self.anticipation_period = anticipation_period
        self.early_period = early_period
        self.inertia_delay = inertia_delay
        self.recall_measure = recall_measure
        self.recall_score = None
        self.false_alarm_score = None
        self.alarm_score = None
        self.anticipation_score = None
        self.density_factor = density_factor

    def _range_by_density(self):
        # TODO
        return self.predictions

    def _group_contiguous_anomalies(self, x):
        changes_index = np.concatenate([[0], np.where(np.diff(x) != 0)[0] + 1])
        anomalous_areas = list()
        for k_change, t_change in enumerate(changes_index):
            if x[t_change] == self.pos_label:
                if k_change < len(changes_index) - 1:
                    anomalous_areas.append(
                        (
                            t_change,
                            changes_index[k_change + 1],
                        )
                    )
                else:
                    anomalous_areas.append(
                        (
                            t_change,
                            len(x),
                        )
                    )
        return anomalous_areas

    def _aggregate_alarms(self, pred):
        return pred

    def _create_fuzzy_mask(self):
        mask = np.ones(len(self.labels), dtype=bool)
        for area_bounds in self.anomalous_areas:
            ## Anticipation mask
            if self.anticipation_period == "default":
                local_anticipation = max((area_bounds[1] - area_bounds[0]) // 5, 1)
            else:
                local_anticipation = self.anticipation_period
            mask[
                max(area_bounds[0] - local_anticipation, 0) : max(area_bounds[0], 0)
            ] = False
            ## Inertia mask
            if (self.inertia_delay > 0) and (area_bounds[1] + 1 != len(self.labels)):
                mask[
                    area_bounds[1]
                    + 1 : min(area_bounds[1] + self.inertia_delay, len(self.labels))
                ] = False

        mask[self.labels == self.pos_label] = False
        return mask

    def _compute_recall_score(self, pred):
        recall_score = np.zeros(len(self.anomalous_areas))
        for i_area, area_bounds in enumerate(self.anomalous_areas):
            recall_score[i_area] = self.recall_measure(
                self.labels[area_bounds[0] : area_bounds[1]],
                pred[area_bounds[0] : area_bounds[1]],
                pos_label=self.pos_label,
            )
        return recall_score

    def _compute_false_alarm_score(self, pred):
        fuzzy_mask = self._create_fuzzy_mask()
        if not sum(fuzzy_mask):
            return 1
        fuzzy_predictions = pred[fuzzy_mask]
        return 1 - sum(fuzzy_predictions == self.pos_label) / sum(fuzzy_mask)

    def _compute_alarm_score(self, pred):
        n_gt = len(self.anomalous_areas)
        n_pred = len(self._group_contiguous_anomalies(pred))
        if n_pred == 0:
            if n_gt == 0:
                return 1
            else:
                return 0
        elif n_gt == 0:
            return 1 / n_pred
        else:
            return min(n_gt / n_pred, n_pred / n_gt)

    def _compute_anticipation_score(self, pred):
        anticipation_score = np.zeros(len(self.anomalous_areas))
        for i_area, area_bounds in enumerate(self.anomalous_areas):
            ## Anticipation
            if area_bounds[0] > 0:
                if self.anticipation_period == "default":
                    start = max(
                        area_bounds[0]
                        - max((area_bounds[1] - area_bounds[0]) // 20, 1),
                        0,
                    )
                else:
                    start = max(area_bounds[0] - self.anticipation_period, 0)
            ## Earliness
            if self.early_period == "default":
                end = min(
                    area_bounds[0] + max((area_bounds[1] - area_bounds[0]) // 10, 1),
                    area_bounds[1],
                )

            else:
                end = min(area_bounds[0] + self.early_period, area_bounds[1])

            local_predictions = pred[start:end]
            weights = special.expit(np.linspace(6, -6, num=len(local_predictions)))
            anticipation_score[i_area] = (
                weights[local_predictions == self.pos_label].sum() / weights.sum()
            )

        return anticipation_score

    def _compute_weighted_score(self):
        weighted_score = np.average(
            a=[
                self.recall_score,
                self.false_alarm_score,
                self.alarm_score,
                self.anticipation_score,
            ],
            weights=[
                self.recall_weight,
                self.false_alarm_weight,
                self.alarm_weight,
                self.anticipation_weight,
            ],
        )
        if self.recall_score == 0:
            return -weighted_score
        return weighted_score

    def compute_metrics(self, labels, predictions, pos_label, return_avg=True):
        self.labels = copy.deepcopy(labels)
        self.predictions = copy.deepcopy(predictions)
        self.predictions = self._range_by_density()
        self.pos_label = pos_label

        self.anomalous_areas = self._group_contiguous_anomalies(self.labels)

        self.raw_recall_score = self._compute_recall_score(pred=self.predictions)
        self.recall_score = np.mean(self.raw_recall_score)
        self.false_alarm_score = self._compute_false_alarm_score(pred=self.predictions)
        self.raw_alarm_score = self._compute_alarm_score(pred=self.predictions)
        self.aggregated_alarm_score = self._compute_alarm_score(
            pred=self._aggregate_alarms(self.predictions)
        )
        self.alarm_score = np.mean([self.raw_alarm_score, self.aggregated_alarm_score])
        self.raw_anticipation_score = self._compute_anticipation_score(
            pred=self.predictions
        )
        self.anticipation_score = np.mean(self.raw_anticipation_score)
        if return_avg:
            return self._compute_weighted_score()
        return np.array(
            [
                self.recall_score,
                self.false_alarm_score,
                self.alarm_score,
                self.anticipation_score,
            ]
        )
