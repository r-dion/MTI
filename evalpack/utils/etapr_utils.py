import time
import datetime
import pandas as pd
import numpy as np
import cv2 as cv
from copy import deepcopy
import pathlib


# To store a single anomaly
class Range:
    def __init__(self, first, last, name):
        self._first_timestamp = first
        self._last_timestamp = last
        self._name = name

    def set_time(self, first, last):
        self._first_timestamp = first
        self._last_timestamp = last

    def get_time(self):
        return self._first_timestamp, self._last_timestamp

    def set_name(self, str):
        self._name = str

    def get_name(self):
        return self._name

    def get_len(self):
        return self._last_timestamp - self._first_timestamp + 1

    def __eq__(self, other):
        return (
            self._first_timestamp == other.get_time()[0]
            and self._last_timestamp == other.get_time()[1]
        )

    def distance(self, other_range) -> int:
        if (
            min(self._last_timestamp, other_range.get_time()[1])
            - max(self._first_timestamp, other_range.get_time()[0])
            > 0
        ):
            return 0
        else:
            return min(
                abs(self._first_timestamp - other_range.get_time()[1]),
                abs(self._last_timestamp - other_range.get_time()[0]),
            )

    def compare(self, other_range) -> int:
        if (
            min(self._last_timestamp, other_range.get_time()[1])
            - max(self._first_timestamp, other_range.get_time()[0])
            > 0
        ):
            return 0
        elif self._last_timestamp - other_range.get_time()[0] < 0:
            return -1
        else:
            return 1


def stream_2_ranges(self, prediction_stream: list) -> list:
    result = []
    for i in range(len(prediction_stream) - 1):
        start_time = 0
        if prediction_stream[i] == 0 and prediction_stream[i + 1] == 1:
            start_time = i + 1
        elif prediction_stream[i] == 1 and prediction_stream[i + 1] == 0:
            result.append(Range(start_time, i, ""))
    return result


def load_stream_2_range(
    stream_data: list, normal_label: int, anomaly_label: int, is_range_name: bool
) -> list:
    return_list = []
    start_id = -1
    end_id = -1
    id = 0
    range_id = 1

    prev_val = -2  # Set prev_val as a value different to normal and anomalous labels

    for val in stream_data:
        if val == anomaly_label and (
            prev_val == normal_label or prev_val == -2
        ):  # Enter the anomaly range
            start_id = id
        elif (
            val == normal_label and prev_val == anomaly_label
        ):  # Go out the anomaly range
            name_buf = ""
            if is_range_name:
                name_buf = str(range_id)
            end_id = id - 1
            return_list.append(Range(start_id, end_id, name_buf))
            range_id += 1
            # start_id = 0

        id += 1
        prev_val = val
    if (
        start_id > end_id
    ):  # start_id != 0 and start_id != -1: #if an anomaly continues till the last point
        return_list.append(Range(start_id, id - 1, str(range_id)))

    return return_list


def load_stream_file(
    filename: str, normal_label: int, anomaly_label: int, is_range_name: bool
) -> list:
    return_list = []
    start_id = -1
    end_id = -1
    id = 0
    range_id = 1
    # is_first = True

    prev_val = -2  # Set prev_val as a value different to normal and anomalous labels

    f = open(filename, "r", encoding="utf-8", newline="")

    for line in f.readlines():
        val = int(line.strip().split()[0])

        """
        #skip the first line
        if is_first:
            if val == anomaly_label:
                start_id = id
            prev_val = val
            is_first = False
            continue
        """

        if val == anomaly_label and (
            prev_val == normal_label or prev_val == -2
        ):  # Enter the anomaly range
            start_id = id
        elif (
            val == normal_label and prev_val == anomaly_label
        ):  # Go out the anomaly range
            name_buf = ""
            if is_range_name:
                name_buf = str(range_id)
            end_id = id - 1
            return_list.append(Range(start_id, end_id, name_buf))
            range_id += 1
            # start_id = 0

        id += 1
        prev_val = val
    f.close()
    if (
        start_id > end_id
    ):  # start_id != 0 and start_id != -1: #if an anomaly continues till the last point
        return_list.append(Range(start_id, id - 1, str(range_id)))

    return return_list


def load_range_file(filename: str, time_format: str) -> list:
    return_list = []
    # is_first = True

    f = open(filename, "r", encoding="utf-8", newline="")
    for line in f.readlines():
        # skip the first line
        # if is_first:
        # is_first = False
        # continue

        items = line.strip().split(",")
        if time_format == "index":
            first_idx = int(items[0])
            last_idx = int(items[1])
        else:
            first_idx = string_to_unixtime(items[0], time_format)
            last_idx = string_to_unixtime(items[1], time_format)

        name_buf = ""
        if len(items) > 2:
            name_buf = str(items[2])

        return_list.append(Range(first_idx, last_idx, name_buf))
    f.close()

    for idx in range(1, len(return_list)):
        if return_list[idx].get_time()[0] <= return_list[idx - 1].get_time()[1]:
            print(
                "Error: ranges ({},{}) and ({},{}) are overlapped in {}".format(
                    return_list[idx - 1].get_time()[0],
                    return_list[idx - 1].get_time()[1],
                    return_list[idx].get_time()[0],
                    return_list[idx].get_time()[1],
                    filename,
                )
            )
            exit(0)

    return return_list


def unixtime_to_string(epoch: int, format: str) -> str:
    return datetime.datetime.fromtimestamp(epoch).strftime(
        format
    )  #'%Y-%m-%d %I:%M:%S %p'


def string_to_unixtime(timestamp: str, format: str) -> int:
    return int(time.mktime(datetime.datetime.strptime(timestamp, format).timetuple()))


def save_range_list(filename: str, range_list: list) -> None:
    f = open(filename, encoding="utf-8", mode="w")
    for single_range in range_list:
        first, last = single_range.get_time()
        f.writelines(
            str(first) + "," + str(last) + "," + single_range.get_name() + "\n"
        )
    f.close()


# Assume that the first line of input files including the information of file format and its corresponding information
# This function handles three types of file format
def load_file(filename: str, filetype: str) -> list:
    assert filetype == "range" or filetype == "stream"

    if filetype == "stream":
        return load_stream_file(filename, 1, -1, True)
    elif filetype == "range":
        return load_range_file(filename, "index")


def make_attack_file(
    input_files: list,
    sep: str,
    label_featname: str,
    input_normal_label: int,
    input_anomalous_label: int,
    output_stream_file: str,
    output_range_file: str,
    output_normal_label: int,
    output_anomalous_label: int,
) -> None:
    label = []
    for an_input_file in input_files:
        temp_file = pd.read_csv(an_input_file, sep=sep)
        label += temp_file[label_featname].values.tolist()

    with open(output_stream_file, "w") as f:
        for a_label in label:
            if a_label == input_normal_label:
                f.write("{}\n".format(output_normal_label))
            elif a_label == input_anomalous_label:
                f.write("{}\n".format(output_anomalous_label))
            else:
                print("There is an unknown label, " + a_label, flush=True)
                f.close()
                return

    ranges = load_stream_2_range(label, 0, 1, False)
    save_range_list(output_range_file, ranges)


def save_range_2_stream(
    filename: str,
    range_list: list,
    last_idx: int,
    normal_label: int,
    anomalous_label: int,
) -> None:
    f = open(filename, encoding="utf-8", mode="w")
    range_id = 0
    for idx in range(last_idx):
        if idx < range_list[range_id].get_time()[0]:
            f.writelines("{}\n".format(normal_label))
        elif (
            range_list[range_id].get_time()[0]
            <= idx
            <= range_list[range_id].get_time()[1]
        ):
            f.writelines("{}\n".format(anomalous_label))
        else:
            f.writelines("{}\n".format(normal_label))
            if range_id < len(range_list) - 1:
                range_id += 1
    f.close()


def convert_index(org_index, max_index, graph_width, margin_left):
    return round(float(org_index / max_index) * graph_width + margin_left)


def draw_csv(
    ranges, img, h_floor, h_ceiling, color, max_index, graph_width, margin_left
):
    for a_range in ranges:
        start_time = convert_index(
            a_range.get_time()[0], max_index, graph_width, margin_left
        )
        end_time = convert_index(
            a_range.get_time()[1], max_index, graph_width, margin_left
        )
        cv.rectangle(
            img, (start_time, h_floor), (end_time, h_ceiling), color, thickness=-1
        )


def draw_csv_range(ranges, img, h_floor, h_ceiling, color, start, end):
    for a_range in ranges:
        if a_range.get_time()[0] <= end or a_range.get_time()[1] >= start:
            cv.rectangle(
                img,
                (a_range.get_time()[0] - start + 10, h_floor),
                (a_range.get_time()[1] - start + 10, h_ceiling),
                color,
                thickness=-1,
            )


def shift_ranges(ranges, first_idx):
    for a_range in ranges:
        a_range.set_time(
            a_range.get_time()[0] - first_idx, a_range.get_time()[1] - first_idx
        )


def draw_graphs(anomalies, predictions, how_show: str):
    method_list = ["Anomalies", "Predictions"]
    anomalies = deepcopy(anomalies)
    predictions = deepcopy(predictions)
    first_idx = min(anomalies[0].get_time()[0] - 100, predictions[0].get_time()[0])
    last_idx = max(anomalies[-1].get_time()[1], predictions[-1].get_time()[1])
    marginal_idx = int(float(last_idx - first_idx) / 100)
    first_idx -= marginal_idx
    shift_ranges(anomalies, first_idx)
    shift_ranges(predictions, first_idx)
    ranges_list = [anomalies, predictions]
    max_index = (
        max(anomalies[-1].get_time()[1], predictions[-1].get_time()[1]) + marginal_idx
    )

    color_list = [
        (70, 70, 70),  # black
        (60, 76, 203),  # red
        (193, 134, 46),  # blue
        (133, 160, 22),  # green
        (206, 143, 187),  # purple
        (94, 73, 52),  # darkblue
        (63, 208, 244),  # yellow
    ]

    margin_left = 10
    margin_right = 150
    margin_top = 20
    margin_bottom = 40

    graph_gap = 20
    graph_height = 40
    graph_width = 2000

    n_results = 2

    width = margin_left + graph_width + margin_right
    height = margin_top + margin_bottom + n_results * (graph_gap + graph_height)
    bpp = 3

    img = np.ones((height, width, bpp), np.uint8) * 255

    img_h = img.shape[0]
    img_w = img.shape[1]
    img_bpp = img.shape[2]

    thickness = 1
    fontsize = 1
    cv.line(
        img,
        (int(margin_left / 2), img_h - margin_bottom),
        (img_w - int(margin_left / 2), img_h - margin_bottom),
        color_list[0],
        thickness,
    )  # x-axis
    pts = np.array(
        [
            [img_w - int(margin_left / 2), img_h - margin_bottom],
            [img_w - int(margin_left / 2) - 7, img_h - margin_bottom + 5],
            [img_w - int(margin_left / 2) - 7, img_h - margin_bottom - 5],
        ],
        np.int32,
    )  # arrow_head
    pts = pts.reshape((-1, 1, 2))
    cv.fillPoly(img, [pts], color_list[0])
    cv.putText(
        img,
        "Relative Index",
        (img_w - 180, img_h - 15),
        cv.FONT_HERSHEY_COMPLEX_SMALL,
        fontsize,
        color_list[0],
        1,
        cv.LINE_AA,
    )  # x-axis label

    for i in range(margin_left, width - margin_right, int(graph_width / 10)):
        cv.line(
            img,
            (i, img_h - margin_bottom + 2),
            (i, img_h - margin_bottom - 2),
            color_list[0],
            thickness,
        )
        org_index = str(round((i - 10) / graph_width * max_index / 1000))
        cv.putText(
            img,
            org_index + "K",
            (i - len(org_index) * 5, img_h - margin_bottom + 25),
            cv.FONT_HERSHEY_COMPLEX_SMALL,
            fontsize,
            color_list[0],
            1,
            cv.LINE_AA,
        )

    thickness = -1
    for idx in range(n_results):
        cv.putText(
            img,
            method_list[idx],
            (
                width - margin_right + 2,
                img_h - margin_bottom - graph_gap * (idx + 1) - graph_height * idx - 12,
            ),
            cv.FONT_HERSHEY_COMPLEX_SMALL,
            fontsize,
            color_list[0],
            1,
            cv.LINE_AA,
        )
        draw_csv(
            ranges_list[idx],
            img,
            h_floor=img_h - margin_bottom - graph_gap * (idx + 1) - graph_height * idx,
            h_ceiling=img_h
            - margin_bottom
            - graph_gap * (idx + 1)
            - graph_height * (idx + 1),
            color=color_list[(idx + 1) % len(color_list)],
            max_index=max_index,
            graph_width=graph_width,
            margin_left=margin_left,
        )

    if how_show == "screen" or how_show == "all":
        cv.imshow("drawing", img)
    if how_show == "file" or how_show == "all":
        cv.imwrite("../../brief_result.png", img)
    if how_show != "screen" and how_show != "all" and how_show != "file":
        print("Parameter Error")
    cv.waitKey(0)


def draw_multi_graphs(
    anomalies, predictions_list, predictions_name_list, how_show: str
):
    method_list = ["Anomalies"] + predictions_name_list

    anomalies = deepcopy(anomalies)
    predictions_list = deepcopy(predictions_list)

    first_idx = anomalies[0].get_time()[0] - 100
    last_idx = anomalies[-1].get_time()[1]
    for single_prediction in predictions_list:
        first_idx = min(first_idx, single_prediction[0].get_time()[0])
        last_idx = max(last_idx, single_prediction[-1].get_time()[1])

    marginal_idx = int(float(last_idx - first_idx) / 100)
    first_idx -= marginal_idx
    shift_ranges(anomalies, first_idx)
    for single_prediction in predictions_list:
        shift_ranges(single_prediction, first_idx)

    ranges_list = [anomalies] + predictions_list

    max_index = anomalies[-1].get_time()[1]
    for single_prediction in predictions_list:
        max_index = max(max_index, single_prediction[-1].get_time()[1])
    max_index = max_index + marginal_idx

    color_list = [
        (0, 0, 0),  # black
        (60, 76, 203),  # red
        (193, 134, 46),  # blue
        (133, 160, 22),  # green
        (206, 143, 187),  # purple
        (94, 73, 52),  # darkblue
        (63, 208, 244),  # yellow
    ]

    margin_left = 10
    margin_right = 180
    margin_top = 20
    margin_bottom = 40

    graph_gap = 20
    graph_height = 40
    graph_width = 2000

    n_results = len(ranges_list)

    width = margin_left + graph_width + margin_right
    height = margin_top + margin_bottom + n_results * (graph_gap + graph_height)
    bpp = 3

    img = np.ones((height, width, bpp), np.uint8) * 255

    img_h = img.shape[0]
    img_w = img.shape[1]
    img_bpp = img.shape[2]

    thickness = 1
    fontsize = 1.4
    cv.line(
        img,
        (int(margin_left / 2), img_h - margin_bottom),
        (img_w - int(margin_left / 2), img_h - margin_bottom),
        color_list[0],
        thickness,
    )  # x-axis
    pts = np.array(
        [
            [img_w - int(margin_left / 2), img_h - margin_bottom],
            [img_w - int(margin_left / 2) - 7, img_h - margin_bottom + 5],
            [img_w - int(margin_left / 2) - 7, img_h - margin_bottom - 5],
        ],
        np.int32,
    )  # arrow_head
    pts = pts.reshape((-1, 1, 2))
    cv.fillPoly(img, [pts], color_list[0])
    cv.putText(
        img,
        "Relative Index",
        (img_w - 180, img_h - 15),
        cv.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        color_list[0],
        1,
        cv.LINE_AA,
    )  # x-axis label

    for i in range(margin_left, width - margin_right, int(graph_width / 10)):
        cv.line(
            img,
            (i, img_h - margin_bottom + 2),
            (i, img_h - margin_bottom - 2),
            color_list[0],
            thickness,
        )
        org_index = str(round((i - 10) / graph_width * max_index / 1000))
        cv.putText(
            img,
            org_index + "K",
            (i - len(org_index) * 5, img_h - margin_bottom + 25),
            cv.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            color_list[0],
            1,
            cv.LINE_AA,
        )

    thickness = -1
    for idx in range(n_results):
        cv.putText(
            img,
            method_list[idx],
            (
                width - margin_right + 2,
                img_h - margin_bottom - graph_gap * (idx + 1) - graph_height * idx - 12,
            ),
            cv.FONT_HERSHEY_COMPLEX_SMALL,
            fontsize,
            color_list[0],
            1,
            cv.LINE_AA,
        )
        draw_csv(
            ranges_list[idx],
            img,
            h_floor=img_h - margin_bottom - graph_gap * (idx + 1) - graph_height * idx,
            h_ceiling=img_h
            - margin_bottom
            - graph_gap * (idx + 1)
            - graph_height * (idx + 1),
            color=color_list[(idx + 1) % len(color_list)],
            max_index=max_index,
            graph_width=graph_width,
            margin_left=margin_left,
        )

    if how_show == "screen" or how_show == "all":
        cv.imshow("drawing", img)
    if how_show == "file" or how_show == "all":
        print("The file is saved at " + str(pathlib.Path(__file__).parent.absolute()))
        cv.imwrite("./brief_result.png", img)
    if how_show != "screen" and how_show != "all" and how_show != "file":
        print("Parameter Error")
    cv.waitKey(0)
