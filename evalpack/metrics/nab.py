import os
import pandas
from evalpack.utils.nab_utils import Sweeper
from evalpack.utils import utils
import numpy as np


def scoreDataSet(args):
    """Function called to score each dataset in the corpus.

    @param args   (tuple)  Arguments to get the detection score for a dataset.

    @return       (tuple)  Contains:
      detectorName  (string)  Name of detector used to get anomaly scores.

      profileName   (string)  Name of profile used to weight each detection type.
                              (tp, tn, fp, fn)

      relativePath  (string)  Path of dataset scored.

      threshold     (float)   Threshold used to convert anomaly scores to
                              detections.

      score         (float)   The score of the dataset.

      counts, tp    (int)     The number of true positive records.

      counts, tn    (int)     The number of true negative records.

      counts, fp    (int)     The number of false positive records.

      counts, fn    (int)     The number of false negative records.

      total count   (int)     The total number of records.
    """
    (
        detectorName,
        profileName,
        relativePath,
        outputPath,
        threshold,
        timestamps,
        anomalyScores,
        windows,
        costMatrix,
        probationaryPercent,
        scoreFlag,
    ) = args

    scorer = Sweeper(probationPercent=probationaryPercent, costMatrix=costMatrix)

    (scores, bestRow) = scorer.scoreDataSet(
        timestamps,
        anomalyScores,
        windows,
        relativePath,
        threshold,
    )

    if scoreFlag:
        # Append scoring function values to the respective results file
        dfCSV = pandas.read_csv(outputPath, header=0, parse_dates=[0])
        dfCSV["S(t)_%s" % profileName] = scores
        dfCSV.to_csv(outputPath, index=False)

    return (
        detectorName,
        profileName,
        relativePath,
        threshold,
        bestRow.score,
        bestRow.tp,
        bestRow.tn,
        bestRow.fp,
        bestRow.fn,
        bestRow.total,
    )


def scaled_sigmoid(y, Atp, Afp, c=5, implemented_version=True):
    if implemented_version:
        coef = 2
    else:
        coef = Atp - Afp
    return coef * (1 / (1 + np.exp(c * y))) - 1


def isin_area(pos, summary):
    if pos >= summary[0] and pos < summary[1]:
        return True
    return False


def get_point_score(pos, summary_anomalous_areas, Atp, Afp, implemented_version):
    for i_area in range(summary_anomalous_areas.shape[0]):
        if isin_area(pos, summary_anomalous_areas[i_area, :]):
            start = summary_anomalous_areas[i_area, 0]
            length = summary_anomalous_areas[i_area, 2]
            return [
                "tp",
                i_area,
                scaled_sigmoid(
                    -1 + (pos - start) / length,
                    Atp=Atp,
                    Afp=Afp,
                    implemented_version=implemented_version,
                ),
            ]
        elif i_area < summary_anomalous_areas.shape[0] - 1:
            if isin_area(
                pos,
                [
                    summary_anomalous_areas[i_area, 1],
                    summary_anomalous_areas[i_area + 1, 0],
                ],
            ):
                end = summary_anomalous_areas[i_area, 1]
                length = summary_anomalous_areas[i_area, 2]
                return [
                    "fp",
                    i_area,
                    scaled_sigmoid(
                        (pos - end + 1) / length,
                        Atp=Atp,
                        Afp=Afp,
                        implemented_version=implemented_version,
                    ),
                ]
        else:
            end = summary_anomalous_areas[-1, 1]
            length = summary_anomalous_areas[-1, 2]
            return [
                "fp",
                i_area,
                scaled_sigmoid(
                    -1 + (pos - end + 1) / length,
                    Atp=Atp,
                    Afp=Afp,
                    implemented_version=implemented_version,
                ),
            ]


def nab_score(
    y_true,
    y_pred,
    Atp=1,
    Afp=0.11,
    Afn=1,
    pos_label=1,
    implemented_version=True,
    return_sum=True,
):
    summary_anomalous_areas = utils.get_anomalous_areas(y_true, pos_label=pos_label)
    tp_fn_score = -np.ones(summary_anomalous_areas.shape[0])
    raw_fp_score = 0
    for i_pred, pred in enumerate(y_pred):
        if pred == pos_label:
            if i_pred < summary_anomalous_areas[0, 0]:
                raw_fp_score += -1
            else:
                res_point_score = get_point_score(
                    i_pred, summary_anomalous_areas, Atp, Afp, implemented_version
                )
                if (res_point_score[0]) == "tp":
                    if res_point_score[2] > tp_fn_score[res_point_score[1]]:
                        tp_fn_score[res_point_score[1]] = res_point_score[2]
                else:
                    raw_fp_score += res_point_score[2]
    if return_sum:
        tp_score = tp_fn_score[tp_fn_score > -1].sum() * Atp
        fn_score = tp_fn_score[tp_fn_score == -1].sum() * Afn
        fp_score = raw_fp_score * Afp
        return tp_score + fn_score + fp_score
    else:
        tp_score = tp_fn_score[tp_fn_score > -1].sum()
        fn_score = tp_fn_score[tp_fn_score == -1].sum()
        fp_score = raw_fp_score
        return np.array([tp_score, fn_score, fp_score])
