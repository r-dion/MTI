import copy
import numpy as np
from sklearn import metrics
from scipy import special


class MTI:
    def __init__(
        self,
        anticipation_weight=1,
        recall_weight=1,
        masked_specificity_weight=1,
        alarm_cardinality_weight=1,
        anticipation_period="default",
        earliness_period="default",
        inertia_delay=100,
        recall_measure=metrics.recall_score,
        density_factor=100,
    ):
        """ "

        Parameters
        ----------
        anticipation_weight: int, default=1
            The Recall score weighting in the final averaging
        recall_weight: int, default=1
            The Recall score weighting in the final averaging
        masked_specificity_weight: int, default=1
            The Masked Specificity score weighting in the final averaging
        alarm_cardinality_weight: int, default=1
            The Alarm Cardinality score weighting in the final averaging
        anticipation_period: str or int, default="default"
            The duration (in number of timestamps) of the anticipation period before an anomaly for the computation of
            the Anticipation/Earliness score and the Masked Specificity score. Default value is "default", which is,
            for each anomalous area, 5% of its length.
        earliness_period: str or int, default="default"
            The duration (in number of timestamps) of the earliness period at the start of an anomaly for the computation of
            the Anticipation/Earliness score. Default value is "default", which is,
            for each anomalous area, 10% of its length.
        inertia_delay: int, default=100
            The duration (in number of timestamps) of the inertia period after an anomaly for the computatio of the
            Masked Specificity score.
        recall_measure: function, default=[sklearn.]metrics.recall_score
            Recall function used for the Recall score component. Default is the traditional one, with the Sklearn implementation.
        density_factor: int, default=100
            # TODO
        """
        self.anticipation_weight = anticipation_weight
        self.recall_weight = recall_weight
        self.masked_specificity_weight = masked_specificity_weight
        self.alarm_cardinality_weight = alarm_cardinality_weight
        self.anticipation_period = anticipation_period
        self.early_period = earliness_period
        self.inertia_delay = inertia_delay
        self.recall_measure = recall_measure
        self.recall_score = None
        self.masked_specificity_score = None
        self.alarm_cardinality_score = None
        self.anticipation_score = None
        self.density_factor = density_factor

    def _range_by_density(self):
        # TODO
        return self.predictions

    def _aggregate_alarms(self, pred):
        # TODO
        return pred

    def _group_contiguous_anomalies(self, x):
        """Constructs anomalous ranges from the discrete labels

        Parameters
        ----------
        x: np.array
            An array of labels

        Returns
        ----------
        anomalous_areas: list
            A list of tuples describing the start and end points of the anomalous ranges.
        """
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

    def _create_fuzzy_mask(self):
        """Constructs the Anticipation and Inertia mask, as a step for the Masked Specificity score.

        Returns
        ----------
        mask: np.array
            An array of booleans that hide the Anticipation area, the Inertia area and the ground truth anomalous ranges.
        """
        mask = np.ones(len(self.labels), dtype=bool)
        for area_bounds in self.anomalous_areas:
            ## Anticipation mask
            if self.anticipation_period == "default":
                local_anticipation = max((area_bounds[1] - area_bounds[0]) // 20, 1)
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
        """

        Parameters
        ----------
        pred: np.array
            An array of the model predictions as binary output (inlier/outlier)

        Returns
        ----------
        recall_score: np.array
            An array of each anomalous range Recall score.
        """
        recall_score = np.zeros(len(self.anomalous_areas))
        for i_area, area_bounds in enumerate(self.anomalous_areas):
            recall_score[i_area] = self.recall_measure(
                self.labels[area_bounds[0] : area_bounds[1]],
                pred[area_bounds[0] : area_bounds[1]],
                pos_label=self.pos_label,
            )
        return recall_score

    def _compute_masked_specificity_score(self, pred):
        """

        Parameters
        ----------
        pred: np.array
            An array of the model predictions as binary output (inlier/outlier)

        Returns
        ----------
        specificity_score: float
            The Masked Specificity score.
        """
        # TODO : replace by input Specificity function
        fuzzy_mask = self._create_fuzzy_mask()
        if not sum(fuzzy_mask):
            return 1
        fuzzy_predictions = pred[fuzzy_mask]
        specificity_score = 1 - sum(fuzzy_predictions == self.pos_label) / sum(
            fuzzy_mask
        )
        return specificity_score

    def _compute_alarm_cardinality_score(self, pred):
        """

        Parameters
        ----------
        pred: np.array
            An array of the model predictions as binary output (inlier/outlier)

        Returns
        ----------
        alarm_cardinality_score: float
            The Alarm Cardinality score.
        """
        n_gt = len(self.anomalous_areas)
        n_pred = len(self._group_contiguous_anomalies(pred))
        if n_pred == 0:
            if n_gt == 0:
                alarm_cardinality_score = 1
            else:
                alarm_cardinality_score = 0
        elif n_gt == 0:
            alarm_cardinality_score = 1 / n_pred
        else:
            alarm_cardinality_score = min(n_gt / n_pred, n_pred / n_gt)
        return alarm_cardinality_score

    def _compute_anticipation_score(self, pred):
        """

        Parameters
        ----------
        pred: np.array
            An array of the model predictions as binary output (inlier/outlier)

        Returns
        ----------
        anticipation_score: np.array
            An array of each anomalous range Anticipation/Earliness score.
        """
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
        """
        Returns
        ----------
        weighted_score: float
            Using the respective weights, the average of the four component scores is computed.
        """
        weighted_score = np.average(
            a=[
                self.recall_score,
                self.masked_specificity_score,
                self.alarm_cardinality_score,
                self.anticipation_score,
            ],
            weights=[
                self.recall_weight,
                self.masked_specificity_weight,
                self.alarm_cardinality_weight,
                self.anticipation_weight,
            ],
        )
        if self.recall_score == 0:
            return -weighted_score
        return weighted_score

    def compute_metrics(self, labels, predictions, pos_label):
        """

        Parameters
        ----------
        labels: np.array
            An array of the ground truth binary labeling
        predictions: np.array
            An array of the model predictions as binary output (inlier/outlier)
        pos_label: int
            The numerical value of an outlier label.
        Returns
        ----------
        mti_score: float
            The MTI final score.
        """
        self.labels = copy.deepcopy(labels)
        self.predictions = copy.deepcopy(predictions)
        self.predictions = self._range_by_density()
        self.pos_label = pos_label

        self.anomalous_areas = self._group_contiguous_anomalies(self.labels)

        self.raw_recall_score = self._compute_recall_score(pred=self.predictions)
        self.recall_score = np.mean(self.raw_recall_score)
        self.masked_specificity_score = self._compute_masked_specificity_score(
            pred=self.predictions
        )
        self.raw_alarm_cardinality_score = self._compute_alarm_cardinality_score(
            pred=self.predictions
        )
        self.aggregated_alarm_cardinality_score = self._compute_alarm_cardinality_score(
            pred=self._aggregate_alarms(self.predictions)
        )
        self.alarm_cardinality_score = np.mean(
            [self.raw_alarm_cardinality_score, self.aggregated_alarm_cardinality_score]
        )
        self.raw_anticipation_score = self._compute_anticipation_score(
            pred=self.predictions
        )
        self.anticipation_score = np.mean(self.raw_anticipation_score)
        mti_score = self._compute_weighted_score()
        return mti_score
