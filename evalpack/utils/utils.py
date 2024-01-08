import numpy as np
import pandas as pd
import copy


def get_contiguous_region(vals, labels=[0, 1]):
    changes_index = np.where(vals[:-1] != vals[1:])[0]
    regions = {label: list() for label in labels}
    if len(changes_index):
        for i, ind in enumerate(changes_index):
            if i != len(changes_index) - 1:
                regions[vals[ind + 1]].append((ind + 1, changes_index[i + 1] + 1))
            else:
                regions[vals[ind + 1]].append((ind + 1, len(vals)))
        regions[vals[0]].append((0, changes_index[0]))
    else:
        regions[vals[0]].append((0, len(vals)))
    return regions


def _get_point_metrics(y_true, y_pred, pos_label):
    """Compute the TP, FP, TN and FN metrics

    Parameters
    ----------
    ----------
    y_true : np.array
        The Ground Truth labelisation
    y_pred : np.array
        The model predictions

    Returns
    -------
    tuple of float
        Returns a tuple with the metrics.
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == pos_label:
            TP += 1
        if y_pred[i] == pos_label and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] != pos_label:
            TN += 1
        if y_pred[i] != pos_label and y_true[i] != y_pred[i]:
            FN += 1

    return (TP, FP, TN, FN)


def evaluate_metric_on_df(
    df, metric, pos_label=-1, neg_label=1, best_fpr_kind=False, max_fpr=0.1
):
    pos_index = df["labels"].values == -1
    neg_index = df["labels"].values == 1
    labels = copy.deepcopy(df["labels"].values)
    labels[pos_index] = pos_label
    labels[neg_index] = neg_label

    columns = df.columns
    columns = columns[columns != "labels"]

    if best_fpr_kind:
        metric_values = np.zeros(columns.size)
        for i_col, col in enumerate(columns):
            thresholds = np.linspace(
                *np.quantile(df[col].values, [0.01, 0.99]), num=100
            )
            thresholds = thresholds[thresholds > 0]
            thresholds_lowFPR = list()
            for threshold in thresholds:
                y_pred_tmp = np.zeros(len(df[col].values))
                y_pred_tmp[df[col].values > threshold] = pos_label
                y_pred_tmp[df[col].values <= threshold] = neg_label
                TP, FP, TN, FN = _get_point_metrics(labels, y_pred_tmp, pos_label)
                fpr = FP / (FP + TN)
                if fpr <= max_fpr:
                    thresholds_lowFPR.append(threshold)

            for i_thresh, threshold in enumerate(thresholds_lowFPR):
                tmp_pred = np.zeros(df[col].values.size)
                tmp_pred[df[col].values > threshold] = pos_label
                tmp_pred[df[col].values <= threshold] = neg_label
                tmp_metric_value = metric(labels, tmp_pred)
                if tmp_metric_value > metric_values[i_col]:
                    metric_values[i_col] = tmp_metric_value

        df_metric = pd.DataFrame(data=metric_values, index=columns)

    else:
        df_metric = pd.DataFrame(
            data=[metric(labels, df[col]) for col in columns], index=columns
        )

    return df_metric
