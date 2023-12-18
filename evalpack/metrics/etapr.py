import argparse
from typing import Callable
import math
import copy
from evalpack.utils import etapr_utils
import numpy as np


class TaPR:
    def __init__(self, theta: float, delta: int):
        self._predictions = []  # list of Ranges
        self._anomalies = []  # list of Ranges
        self._ambiguous_inst = []  # list of Ranges

        self._set_predictions = False
        self._set_anomalies = False

        # self._rho = theta
        # self._pi = theta
        self._theta = theta
        self._delta = delta

        pass

    def set_anomalies(self, anomaly_list: list) -> None:
        self._anomalies = copy.deepcopy(anomaly_list)
        self._gen_ambiguous()
        self._set_anomalies = True

    def set_predictions(self, prediction_list: list) -> None:
        self._predictions = copy.deepcopy(prediction_list)
        self._set_predictions = True

    def _gen_ambiguous(self):
        for i in range(len(self._anomalies)):
            start_id = self._anomalies[i].get_time()[1] + 1
            end_id = start_id + self._delta

            # if the next anomaly occurs during the theta, update the end_id
            if (
                i + 1 < len(self._anomalies)
                and end_id > self._anomalies[i + 1].get_time()[0]
            ):
                end_id = self._anomalies[i + 1].get_time()[0] - 1

            if start_id > end_id:
                start_id = -2
                end_id = -1

            self._ambiguous_inst.append(etapr_utils.Range(start_id, end_id, str(i)))

    def get_n_predictions(self):
        return len(self._predictions)

    def get_n_anomalies(self):
        return len(self._anomalies)

    def _ids_2_objects(self, id_list, range_list):
        result = []
        for id in id_list:
            result.append(range_list[id])
        return result

    def TaR_d(self) -> float and list:
        score, detected_id_list = self._TaR_d(
            self._anomalies, self._ambiguous_inst, self._predictions, self._theta
        )
        return score, self._ids_2_objects(detected_id_list, self._anomalies)

    def _TaR_d(
        self, anomalies: list, ambiguous_inst: list, predictions: list, threshold: float
    ) -> float and list:
        total_score = 0.0
        detected_anomalies = []
        for anomaly_id in range(len(anomalies)):
            anomaly = anomalies[anomaly_id]
            ambiguous = ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(
                anomaly.get_time()[0],
                anomaly.get_time()[1],
                anomaly.get_time()[0],
                anomaly.get_time()[1],
                self._uniform_func,
            )

            score = 0.0
            for prediction in predictions:
                score += self._overlap_and_subsequent_score(
                    anomaly, ambiguous, prediction
                )

            if min(1.0, score / max_score) >= threshold:
                total_score += 1.0
                detected_anomalies.append(anomaly_id)

        if len(anomalies) == 0:
            return 0.0, []
        else:
            return total_score / len(anomalies), detected_anomalies

    def TaP_d(self) -> float and list:
        score, correct_id_list = self._TaP_d(
            self._anomalies, self._ambiguous_inst, self._predictions, self._theta
        )
        return score, self._ids_2_objects(correct_id_list, self._predictions)

    def _TaP_d(self, anomalies, ambiguous_inst, predictions, threshold):
        correct_predictions = []
        total_score = 0.0
        for prediction_id in range(len(predictions)):
            max_score = (
                predictions[prediction_id].get_time()[1]
                - predictions[prediction_id].get_time()[0]
                + 1
            )

            score = 0.0
            for anomaly_id in range(len(anomalies)):
                anomaly = anomalies[anomaly_id]
                ambiguous = ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(
                    anomaly, ambiguous, predictions[prediction_id]
                )

            if (score / max_score) >= threshold:
                total_score += 1.0
                correct_predictions.append(prediction_id)

        if len(predictions) == 0:
            return 0.0, []
        else:
            return total_score / len(predictions), correct_predictions

    def _detect(self, src_range: etapr_utils.Range, ranges: list, theta: int) -> bool:
        rest_len = src_range.get_time()[1] - src_range.get_time()[0] + 1
        for dst_range in ranges:
            len = self._overlapped_len(src_range, dst_range)
            if len != -1:
                rest_len -= len
        return (float)(rest_len) / (
            src_range.get_time()[1] - src_range.get_time()[0] + 1
        ) <= (1.0 - theta)

    def _overlapped_len(
        self, range1: etapr_utils.Range, range2: etapr_utils.Range
    ) -> int:
        detected_start = max(range1.get_time()[0], range2.get_time()[0])
        detected_end = min(range1.get_time()[1], range2.get_time()[1])

        if detected_end < detected_start:
            return 0
        else:
            return detected_end - detected_start + 1

    def _min_max_norm(
        self, value: int, org_min: int, org_max: int, new_min: int, new_max: int
    ) -> float:
        if org_min == org_max:
            return new_min
        else:
            return (float)(new_min) + (float)(value - org_min) * (new_max - new_min) / (
                org_max - org_min
            )

    def _decaying_func(self, val: float) -> float:
        assert -6 <= val <= 6
        return 1 / (1 + math.exp(val))

    def _ascending_func(self, val: float) -> float:
        assert -6 <= val <= 6
        return 1 / (1 + math.exp(val * -1))

    def _uniform_func(self, val: float) -> float:
        return 1.0

    def _sum_of_func(
        self,
        start_time: int,
        end_time: int,
        org_start: int,
        org_end: int,
        func: Callable[[float], float],
    ) -> float:
        val = 0.0
        for timestamp in range(start_time, end_time + 1):
            val += func(self._min_max_norm(timestamp, org_start, org_end, -6, 6))
        return val

    def _overlap_and_subsequent_score(
        self,
        anomaly: etapr_utils.Range,
        ambiguous: etapr_utils.Range,
        prediction: etapr_utils.Range,
    ) -> float:
        score = 0.0

        detected_start = max(anomaly.get_time()[0], prediction.get_time()[0])
        detected_end = min(anomaly.get_time()[1], prediction.get_time()[1])

        score += self._sum_of_func(
            detected_start,
            detected_end,
            anomaly.get_time()[0],
            anomaly.get_time()[1],
            self._uniform_func,
        )

        if ambiguous.get_time()[0] < ambiguous.get_time()[1]:
            detected_start = max(ambiguous.get_time()[0], prediction.get_time()[0])
            detected_end = min(ambiguous.get_time()[1], prediction.get_time()[1])

            score += self._sum_of_func(
                detected_start,
                detected_end,
                ambiguous.get_time()[0],
                ambiguous.get_time()[1],
                self._decaying_func,
            )

        return score

    def TaR_p(self) -> float:
        total_score = 0.0
        for anomaly_id in range(len(self._anomalies)):
            anomaly = self._anomalies[anomaly_id]
            ambiguous = self._ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(
                anomaly.get_time()[0],
                anomaly.get_time()[1],
                anomaly.get_time()[0],
                anomaly.get_time()[1],
                self._uniform_func,
            )

            score = 0.0
            for prediction in self._predictions:
                score += self._overlap_and_subsequent_score(
                    anomaly, ambiguous, prediction
                )

            total_score += min(1.0, score / max_score)

        if len(self._anomalies) == 0:
            return 0.0
        else:
            return total_score / len(self._anomalies)

    def TaP_p(self) -> float:
        total_score = 0.0
        for prediction in self._predictions:
            max_score = prediction.get_time()[1] - prediction.get_time()[0] + 1

            score = 0.0
            for anomaly_id in range(len(self._anomalies)):
                anomaly = self._anomalies[anomaly_id]
                ambiguous = self._ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(
                    anomaly, ambiguous, prediction
                )

            total_score += score / max_score

        if len(self._predictions) == 0:
            return 0.0
        else:
            return total_score / len(self._predictions)


def compute(
    anomalies: list, predictions: list, alpha: float, theta: float, delta: int
) -> dict:
    ev = TaPR(theta, delta)

    ev.set_anomalies(anomalies)
    ev.set_predictions(predictions)

    tard_value, detected_list = ev.TaR_d()
    tarp_value = ev.TaR_p()

    tapd_value, correct_list = ev.TaP_d()
    tapp_value = ev.TaP_p()

    result = {}
    tar_value = alpha * tard_value + (1 - alpha) * tarp_value
    result["TaR"] = tar_value
    result["TaRd"] = tard_value
    result["TaRp"] = tarp_value

    tap_value = alpha * tapd_value + (1 - alpha) * tapp_value
    result["TaP"] = tap_value
    result["TaPd"] = tapd_value
    result["TaPp"] = tapp_value

    detected_anomalies = []
    for value in detected_list:
        detected_anomalies.append(value.get_name())

    result["Detected_Anomalies"] = detected_anomalies
    result["Detected_Anomalies_Ranges"] = detected_list
    result["Correct_Predictions_Ranges"] = correct_list

    if tar_value + tap_value == 0:
        result["f1"] = 0.0
    else:
        result["f1"] = (2 * tar_value * tap_value) / (tar_value + tap_value)

    return result


def compute_with_load(
    anomaly_file: str,
    prediction_file: str,
    file_type: str,
    alpha: float,
    theta: float,
    delta: int,
) -> dict:
    anomalies = etapr_utils.load_file(anomaly_file, file_type)
    predictions = etapr_utils.load_file(prediction_file, file_type)
    return compute(anomalies, predictions, alpha, theta, delta)


def print_result(
    anomalies: list,
    predictions: list,
    alpha: float,
    theta: float,
    delta: int,
    verbose: bool,
    graph: str,
) -> None:
    org_predictions = copy.deepcopy(predictions)
    result = compute(anomalies, predictions, alpha, theta, delta)

    print(
        "The parameters (alpha, theta, delta) are set as %g, %g, and %d."
        % (alpha, theta, delta)
    )

    print("\n[TaR]:", "%0.5f" % result["TaR"])
    print("\t* Detection score:", "%0.5f" % result["TaRd"])
    print("\t* Portion score:", "%0.5f" % result["TaRp"])
    if verbose:
        buf = "\t\tdetected anomalies: "
        if len(result["Detected_Anomalies_Ranges"]) == 0:
            buf += "None  "
        else:
            for value in result["Detected_Anomalies_Ranges"]:
                buf += (
                    value.get_name()
                    + "("
                    + str(value.get_time()[0])
                    + ":"
                    + str(value.get_time()[1])
                    + "), "
                )
        print(buf[:-2])

    print("\n[TaP]:", "%0.5f" % result["TaP"])
    print("\t* Detection score:", "%0.5f" % result["TaPd"])
    print("\t* Portion score:", "%0.5f" % result["TaPp"])
    if verbose:
        buf = "\t\tcorrect predictions: "
        if len(result["Correct_Predictions_Ranges"]) == 0:
            buf += "None  "
        else:
            for value in result["Correct_Predictions_Ranges"]:
                buf += (
                    value.get_name()
                    + "("
                    + str(value.get_time()[0])
                    + ":"
                    + str(value.get_time()[1])
                    + "), "
                )
        print(buf[:-2])

    assert graph == "screen" or graph == "file" or graph == "none" or graph == "all"
    if graph == "screen" or graph == "file" or graph == "all":
        etapr_utils.draw_graphs(anomalies, org_predictions, graph)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--anomalies", help="anomaly file name (ground truth)", required=True
    )
    argument_parser.add_argument(
        "--predictions", help="prediction file name", required=True
    )
    argument_parser.add_argument(
        "--filetype",
        help="choose the file type between range and stream",
        required=True,
    )
    argument_parser.add_argument("--graph", help="show graph of results")

    argument_parser.add_argument(
        "--verbose", help="show detail results", action="store_true"
    )
    argument_parser.add_argument("--theta", help="set parameter theta")
    argument_parser.add_argument("--alpha", help="set parameter alpha")
    argument_parser.add_argument("--delta", help="set parameter delta")
    arguments = argument_parser.parse_args()

    arguments = argument_parser.parse_args()
    theta, alpha, delta, graph = 0.5, 0.8, 600, "none"  # default values
    if arguments.theta is not None:
        theta = float(arguments.theta)
    if arguments.alpha is not None:
        alpha = float(arguments.alpha)
    if arguments.delta is not None:
        delta = int(arguments.delta)
    if arguments.graph is not None:
        graph = arguments.graph

    assert 0.0 <= theta <= 1.0
    assert 0.0 <= alpha <= 1.0
    assert isinstance(delta, int)
    assert graph == "screen" or graph == "file" or graph == "none" or graph == "all"

    anomalies = etapr_utils.load_file(arguments.anomalies, arguments.filetype)
    predictions = etapr_utils.load_file(arguments.predictions, arguments.filetype)

    print_result(anomalies, predictions, alpha, theta, delta, arguments.verbose, graph)


class eTaPR(TaPR):
    def __init__(self, theta_p, theta_r, delta=0.0):
        super(eTaPR, self).__init__(0.0, 0)
        self._predictions_weight = []
        self._predictions_total_weight = 0.0
        self._prune_predictions = []

        self._theta_p = theta_p
        self._theta_r = theta_r
        self._delta_ratio = delta

        self._overlap_score_mat_org = np.zeros(1)
        self._overlap_score_mat_elm = np.zeros(1)  # eleminate by prunning
        self._max_anomaly_score = []
        self._max_prediction_score = []

        self._weight_func = math.sqrt

    def _gen_ambiguous(self):
        for i in range(len(self._anomalies)):
            start_id = self._anomalies[i].get_time()[1] + 1
            end_id = start_id + int(
                self._delta_ratio
                * (self._anomalies[i].get_time()[1] - self._anomalies[i].get_time()[0])
            )

            # if the next anomaly occurs during the theta, update the end_id
            if (
                i + 1 < len(self._anomalies)
                and end_id > self._anomalies[i + 1].get_time()[0]
            ):
                end_id = self._anomalies[i + 1].get_time()[0] - 1

            if start_id > end_id:
                start_id = -2
                end_id = -1

            self._ambiguous_inst.append(etapr_utils.Range(start_id, end_id, str(i)))

    # load data -> build the score matrix -> do pruning
    def set(self, anomalies: list, predictions: list) -> None:
        # loading data
        self.set_anomalies(anomalies)
        self.set_predictions(predictions)

        # computing weights
        for a_prediction in self._predictions:
            first, last = a_prediction.get_time()
            temp_weight = math.sqrt(last - first + 1)
            self._predictions_weight.append(temp_weight)
            self._predictions_total_weight += temp_weight

        # computing the score matrix
        self._overlap_score_mat_org = np.zeros(
            (self.get_n_anomalies(), self.get_n_predictions())
        )
        for anomaly_id in range(self.get_n_anomalies()):
            for prediction_id in range(self.get_n_predictions()):
                self._overlap_score_mat_org[anomaly_id, prediction_id] = float(
                    self._overlap_and_subsequent_score(
                        self._anomalies[anomaly_id],
                        self._ambiguous_inst[anomaly_id],
                        self._predictions[prediction_id],
                    )
                )

        # computing the maximum scores for each anomaly or prediction
        for an_anomaly in self._anomalies:
            start, end = an_anomaly.get_time()
            self._max_anomaly_score.append(
                float(self._sum_of_func(start, end, start, end, self._uniform_func))
            )
        for a_prediction in self._predictions:
            self._max_prediction_score.append(a_prediction.get_len())

        # pruning
        self._pruning()

    def _pruning(self):
        self._overlap_score_mat_elm = self._overlap_score_mat_org.copy()

        while True:
            tars = self._overlap_score_mat_elm.sum(axis=1) / self._max_anomaly_score
            elem_anomaly_ids = set(np.where(tars < self._theta_r)[0]) - set(
                np.where(tars == 0.0)[0]
            )
            for id in elem_anomaly_ids:
                self._overlap_score_mat_elm[id] = np.zeros(self.get_n_predictions())
            taps = self._overlap_score_mat_elm.sum(axis=0) / self._max_prediction_score
            elem_prediction_ids = set(np.where(taps < self._theta_p)[0]) - set(
                np.where(taps == 0.0)[0]
            )
            for id in elem_prediction_ids:
                self._overlap_score_mat_elm[:, id] = np.zeros(self.get_n_anomalies())

            if len(elem_anomaly_ids) == 0 and len(elem_prediction_ids) == 0:
                break

    def _etar_d(self, theta: float) -> np.array and list:
        if self.get_n_anomalies() == 0.0 or self.get_n_predictions() == 0.0:
            return np.zeros(self.get_n_anomalies()), []

        scores = self._overlap_score_mat_elm.sum(axis=1) / self._max_anomaly_score
        scores = np.where(scores >= theta, 1.0, scores)
        scores = np.where(scores < theta, 0.0, scores)
        detected_id_list = np.where(scores >= theta)[0]

        return scores, detected_id_list

    def eTaR_d(self) -> float and list:
        _, detected_id_list = self._etar_d(self._theta_r)
        return len(detected_id_list) / self.get_n_anomalies(), detected_id_list

    def _etar_p(self) -> np.array:
        if self.get_n_anomalies() == 0.0 or self.get_n_predictions() == 0.0:
            return 0.0

        scores = self._overlap_score_mat_elm.sum(axis=1) / self._max_anomaly_score
        scores = np.where(scores > 1.0, 1.0, scores)
        return scores

    def eTaR_p(self) -> float:
        scores = self._etar_p()
        return scores.mean()

    def eTaR(self) -> float:
        detection_scores, _ = self._etar_d(self._theta_r)
        portion_scores = self._etar_p()

        return ((detection_scores + detection_scores * portion_scores) / 2).mean()

    def _etap_d(self, theta: float) -> np.array and list:
        if self.get_n_anomalies() == 0.0 or self.get_n_predictions() == 0.0:
            return 0.0, []

        scores = self._overlap_score_mat_elm.sum(axis=0) / self._max_prediction_score
        scores = np.where(scores >= theta, 1.0, scores)
        scores = np.where(scores < theta, 0.0, scores)
        correct_id_list = np.where(scores >= theta)[0]

        return scores, correct_id_list

    def eTaP_d(self) -> float and list:
        _, correct_id_list = self._etap_d(self._theta_p)

        tapd = 0.0
        for correct_id in correct_id_list:
            tapd += self._predictions_weight[correct_id]
        tapd /= float(self._predictions_total_weight)

        return tapd, correct_id_list

    def _etap_p(self) -> np.array:
        if self.get_n_anomalies() == 0.0 or self.get_n_predictions() == 0.0:
            return 0.0

        scores = self._overlap_score_mat_elm.sum(axis=0) / self._max_prediction_score
        return scores

    def eTaP_p(self) -> float:
        scores = self._etap_p()

        final_score = 0.0
        for i in range(len(scores)):
            final_score += float(self._predictions_weight[i]) * scores[i]
        final_score /= float(self._predictions_total_weight)
        return final_score

    def eTaP(self) -> float:
        detection_scores, _ = self._etap_d(self._theta_p)
        portion_scores = self._etap_p()

        scores = (detection_scores + detection_scores * portion_scores) / 2
        final_score = 0.0
        for i in range(len(scores)):
            final_score += float(self._predictions_weight[i]) * scores[i]
        final_score /= float(self._predictions_total_weight)
        return final_score

    # conventional precision
    def precision(self) -> float:
        if self.get_n_anomalies() == 0.0 or self.get_n_predictions() == 0.0:
            return 0.0

        return self._overlap_score_mat_org.sum() / sum(self._max_prediction_score)

    # conventional recall
    def recall(self) -> float:
        if self.get_n_anomalies() == 0.0 or self.get_n_predictions() == 0.0:
            return 0.0

        return self._overlap_score_mat_org.sum() / sum(self._max_anomaly_score)

    # point adjust precision
    def point_adjust_precision(self, theta: float) -> float:
        if self.get_n_anomalies() == 0.0 or self.get_n_predictions() == 0.0:
            return 0.0

        _, detected_id_list = self._TaR_d(
            self._anomalies,
            [etapr_utils.Range(-2, -1, "") for i in range(len(self._anomalies))],
            self._predictions,
            theta,
        )

        hit_cnt = 0
        for detected_id in detected_id_list:
            hit_cnt += self._anomalies[detected_id].get_len()

        extended_predictions_len = (
            sum(self._max_prediction_score)
            + hit_cnt
            - self._overlap_score_mat_org.sum()
        )

        return hit_cnt / extended_predictions_len

    def point_adjust_recall(self, theta: float) -> float:
        if self.get_n_anomalies() == 0.0 or self.get_n_predictions() == 0.0:
            return 0.0

        _, detected_id_list = self._TaR_d(
            self._anomalies,
            [etapr_utils.Range(-2, -1, "") for i in range(len(self._anomalies))],
            self._predictions,
            theta,
        )
        hit_cnt = 0
        for detected_id in detected_id_list:
            hit_cnt += self._anomalies[detected_id].get_len()
        return hit_cnt / sum(self._max_anomaly_score)


def evaluate_w_ranges(
    anomalies: list,
    predictions: list,
    theta_p: float,
    theta_r: float,
    delta: float = 0.0,
) -> dict:
    assert 0.0 <= theta_p <= 1.0
    assert 0.0 <= theta_r <= 1.0
    assert 0.0 <= delta <= 1.0

    ev = eTaPR(theta_p, theta_r, delta)
    ev.set(anomalies, predictions)

    tard_value, detected_id_list = ev.eTaR_d()
    tarp_value = ev.eTaR_p()
    tar_value = ev.eTaR()

    tapd_value, correct_id_list = ev.eTaP_d()
    tapp_value = ev.eTaP_p()
    tap_value = ev.eTaP()

    result = {}
    result["eTaR"] = tar_value
    result["eTaRd"] = tard_value
    result["eTaRp"] = tarp_value

    result["eTaP"] = tap_value
    result["eTaPd"] = tapd_value
    result["eTaPp"] = tapp_value

    detected_anomalies = []
    for id in detected_id_list:
        detected_anomalies.append(anomalies[id])

    correct_predictions = []
    for id in correct_id_list:
        correct_predictions.append(predictions[id])

    result["Detected_Anomalies"] = detected_anomalies
    result["Correct_Predictions"] = correct_predictions

    if tar_value + tap_value == 0:
        result["f1"] = 0.0
    else:
        result["f1"] = (2 * tar_value * tap_value) / (tar_value + tap_value)

    false_alarm = 0
    false_alarm_cnt = 0
    for id in range(len(predictions)):
        if id not in correct_id_list:
            false_alarm += predictions[id].get_len()
            false_alarm_cnt += 1
    result["False Alarm"] = false_alarm
    result["N False Alarm"] = false_alarm_cnt

    result["precision"] = ev.precision()
    result["recall"] = ev.recall()

    result["point_adjust_precision"] = ev.point_adjust_precision(1e-10)
    result["point_adjust_recall"] = ev.point_adjust_recall(1e-10)

    return result


def evaluate_w_streams(
    anomalies: list,
    predictions: list,
    theta_p=0.7,
    theta_r: float = 0.1,
    delta: float = 0.0,
) -> dict:
    assert 0.0 <= theta_p <= 1.0
    assert 0.0 <= theta_r <= 1.0
    assert 0.0 <= delta <= 1.0

    anomalous_ranges = etapr_utils.load_stream_2_range(anomalies, 0, 1, True)
    predicted_ranges = etapr_utils.load_stream_2_range(predictions, 0, 1, True)

    return evaluate_w_ranges(
        anomalies=anomalous_ranges,
        predictions=predicted_ranges,
        theta_p=theta_p,
        theta_r=theta_r,
        delta=delta,
    )


def evaluate_w_files(
    anomaly_file: str,
    prediction_file: str,
    file_type: str,
    theta_p: float,
    theta_r: float,
    delta: float = 0.0,
) -> dict:
    assert 0.0 <= theta_p <= 1.0
    assert 0.0 <= theta_r <= 1.0
    assert 0.0 <= delta <= 1.0

    anomalies = etapr_utils.load_file(anomaly_file, file_type)
    predictions = etapr_utils.load_file(prediction_file, file_type)

    return evaluate_w_ranges(anomalies, predictions, theta_p, theta_r, delta)


def print_results(result: dict, verbose: bool) -> None:
    print("\n[TaR]:", "%0.5f" % result["TaR"])
    print("\t* Detection score:", "%0.5f" % result["TaRd"])
    print("\t* Portion score:", "%0.5f" % result["TaRp"])
    if verbose:
        buf = "\t\tdetected anomalies: "
        if len(result["Detected_Anomalies"]) == 0:
            buf += "None  "
        else:
            for value in result["Detected_Anomalies"]:
                buf += (
                    value.get_name()
                    + "("
                    + str(value.get_time()[0])
                    + ":"
                    + str(value.get_time()[1])
                    + "), "
                )
        print(buf[:-2])

    print("\n[TaP]:", "%0.5f" % result["TaP"])
    print("\t* Detection score:", "%0.5f" % result["TaPd"])
    print("\t* Portion score:", "%0.5f" % result["TaPp"])
    if verbose:
        buf = "\t\tcorrect predictions: "
        if len(result["Correct_Predictions"]) == 0:
            buf += "None  "
        else:
            for value in result["Correct_Predictions"]:
                buf += (
                    value.get_name()
                    + "("
                    + str(value.get_time()[0])
                    + ":"
                    + str(value.get_time()[1])
                    + "), "
                )
        print(buf[:-2])


def draw_graph(anomalies: list, predictions: list, graph_dst: str) -> None:
    assert (
        graph_dst == "screen"
        or graph_dst == "file"
        or graph_dst == "none"
        or graph_dst == "all"
    )
    if graph_dst == "screen" or graph_dst == "file" or graph_dst == "all":
        etapr_utils.draw_graphs(anomalies, predictions, graph_dst)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--anomalies", help="anomaly file name (ground truth)", required=True
    )
    argument_parser.add_argument(
        "--predictions", help="prediction file name", required=True
    )
    argument_parser.add_argument(
        "--filetype",
        help="choose the file type between range and stream",
        required=True,
    )
    argument_parser.add_argument("--graph", help="show graph of results")

    argument_parser.add_argument(
        "--verbose", help="show detail results", action="store_true"
    )
    argument_parser.add_argument("--theta_r", help="set parameter theta_r")
    argument_parser.add_argument("--theta_p", help="set parameter theta_p")
    argument_parser.add_argument("--delta", help="set parameter delta")
    # arguments = argument_parser.parse_args()

    arguments = argument_parser.parse_args()
    theta_p, theta_r, delta, graph = 0.5, 0.1, 0.0, "none"  # default values
    if arguments.tp is not None:
        theta_p = float(arguments.tp)
    if arguments.tr is not None:
        theta_r = float(arguments.tr)
    if arguments.delta is not None:
        delta = int(arguments.delta)
    if arguments.graph is not None:
        graph = arguments.graph

    # assert(isinstance(delta, int))
    assert graph == "screen" or graph == "file" or graph == "none" or graph == "all"

    anomalies = etapr_utils.load_file(arguments.anomalies, arguments.filetype)
    predictions = etapr_utils.load_file(arguments.predictions, arguments.filetype)
    results = evaluate_w_ranges(anomalies, predictions, theta_p, theta_r, delta)

    print_results(results, arguments.verbose)
    draw_graph(anomalies, predictions, graph)
