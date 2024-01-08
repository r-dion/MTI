from evalpack.utils import affiliation_utils

from evalpack.utils import utils


def test_events(events):
    """
    Verify the validity of the input events
    :param events: list of events, each represented by a couple (start, stop)
    :return: None. Raise an error for incorrect formed or non ordered events
    """
    if type(events) is not list:
        raise TypeError("Input `events` should be a list of couples")
    if not all([type(x) is tuple for x in events]):
        raise TypeError("Input `events` should be a list of tuples")
    if not all([len(x) == 2 for x in events]):
        raise ValueError("Input `events` should be a list of couples (start, stop)")
    if not all([x[0] <= x[1] for x in events]):
        raise ValueError(
            "Input `events` should be a list of couples (start, stop) with start <= stop"
        )
    if not all([events[i][1] < events[i + 1][0] for i in range(len(events) - 1)]):
        raise ValueError("Couples of input `events` should be disjoint and ordered")


def pr_from_events(events_pred, events_gt, Trange):
    """
    Compute the affiliation metrics including the precision/recall in [0,1],
    along with the individual precision/recall distances and probabilities

    :param events_pred: list of predicted events, each represented by a couple
    indicating the start and the stop of the event
    :param events_gt: list of ground truth events, each represented by a couple
    indicating the start and the stop of the event
    :param Trange: range of the series where events_pred and events_gt are included,
    represented as a couple (start, stop)
    :return: dictionary with precision, recall, and the individual metrics
    """
    # testing the inputs
    test_events(events_pred)
    test_events(events_gt)

    # other tests
    minimal_Trange = affiliation_utils.infer_Trange(events_pred, events_gt)
    if not Trange[0] <= minimal_Trange[0]:
        raise ValueError("`Trange` should include all the events")
    if not minimal_Trange[1] <= Trange[1]:
        raise ValueError("`Trange` should include all the events")

    if len(events_gt) == 0:
        raise ValueError("Input `events_gt` should have at least one event")

    if affiliation_utils.has_point_anomalies(
        events_pred
    ) or affiliation_utils.has_point_anomalies(events_gt):
        raise ValueError("Cannot manage point anomalies currently")

    if Trange is None:
        # Set as default, but Trange should be indicated if probabilities are used
        raise ValueError(
            "Trange should be indicated (or inferred with the `infer_Trange` function"
        )

    E_gt = affiliation_utils.get_all_E_gt_func(events_gt, Trange)
    aff_partition = affiliation_utils.affiliation_partition(events_pred, E_gt)

    # Computing precision distance
    d_precision = [
        affiliation_utils.affiliation_precision_distance(Is, J)
        for Is, J in zip(aff_partition, events_gt)
    ]

    # Computing recall distance
    d_recall = [
        affiliation_utils.affiliation_recall_distance(Is, J)
        for Is, J in zip(aff_partition, events_gt)
    ]

    # Computing precision
    p_precision = [
        affiliation_utils.affiliation_precision_proba(Is, J, E)
        for Is, J, E in zip(aff_partition, events_gt, E_gt)
    ]

    # Computing recall
    p_recall = [
        affiliation_utils.affiliation_recall_proba(Is, J, E)
        for Is, J, E in zip(aff_partition, events_gt, E_gt)
    ]

    if affiliation_utils._len_wo_nan(p_precision) > 0:
        p_precision_average = affiliation_utils._sum_wo_nan(
            p_precision
        ) / affiliation_utils._len_wo_nan(p_precision)
    else:
        p_precision_average = p_precision[0]  # math.nan
    p_recall_average = sum(p_recall) / len(p_recall)

    dict_out = dict(
        {
            "precision": p_precision_average,
            "recall": p_recall_average,
            "individual_precision_probabilities": p_precision,
            "individual_recall_probabilities": p_recall,
            "individual_precision_distances": d_precision,
            "individual_recall_distances": d_recall,
        }
    )
    return dict_out


def produce_all_results():
    """
    Produce the affiliation precision/recall for all files
    contained in the `data` repository
    :return: a dictionary indexed by data names, each containing a dictionary
    indexed by algorithm names, each containing the results of the affiliation
    metrics (precision, recall, individual probabilities and distances)
    """
    (
        datasets,
        Tranges,
    ) = affiliation_utils.read_all_as_events()  # read all the events in folder `data`
    results = dict()
    for data_name in datasets.keys():
        results_data = dict()
        for algo_name in datasets[data_name].keys():
            if algo_name != "groundtruth":
                results_data[algo_name] = affiliation_utils.pr_from_events(
                    datasets[data_name][algo_name],
                    datasets[data_name]["groundtruth"],
                    Tranges[data_name],
                )
        results[data_name] = results_data
    return results


def affiliation_f1_score(y_true, y_pred, pos_label):
    event_labels = utils.get_contiguous_region(y_true)[pos_label]
    event_pred = utils.get_contiguous_region(y_pred)[pos_label]
    res = pr_from_events(event_pred, event_labels, (0, y_true.size))
    return affiliation_utils.f1_func(res["precision"], res["recall"])
