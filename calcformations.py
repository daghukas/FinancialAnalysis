"""Detects formations in nasdaq-charts. Current: DoubleTop and DoubleBottom."""
import argparse
import math
import pandas as pd
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

found_necklines = []
found_breaking_of_necklines = []

def main():
    """Set-up argparser and process arguments."""

    # set input parameter and help message
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        "-c",
        metavar="C",
        required=True,
        help="Set the nasdaq company you want to examine. [ident e.g. AMZN for Amazon]")

    my_parser.add_argument(
        "-t",
        metavar="t",
        default="minute",
        choices=["day","minute"],
        help="Set the time frame of each candle in the chart. [choose between day and minute]")

    my_parser.add_argument(
        "-d",
        metavar="d",
        help="Set the date of data set")

    my_parser.add_argument(
        "-f",
        metavar="f",
        help="Set the formations you want to be detected. 0 = Double Top, 1 = Double Bottom")

    args = my_parser.parse_args()

    company = args.c.upper()
    time_frame = args.t.lower()
    chart_data = read_file(company, time_frame)
    prepare_data(chart_data)

def read_file(company, time_frame):
    """Reads file with chart_data."""

    filepath = (r'/data/financedata/2020ss/gelling/data/kibot3/NASDAQ/{company}/{company.lower()}'
        r'.candle.{time_frame}.unadjusted')
    chart_data = pd.read_csv(
        filepath,
        ",",
        header=None,
        names=["Date","Time","Open", "High","Low","Close","Umsatz (St)", "Quelle", "Import-Datum"],
        parse_dates=[0])
    return chart_data

def prepare_data(chart_data):
    """Prepares data for further calculations. """

    chart_data = chart_data[:300]
    #dataset_index = chart_data.index.tolist()
    dataset_close_val = chart_data[['Close']]

    # get indizes of maxs and mins
    arr_index_maxima = sp.argrelextrema(np.array(dataset_close_val), np.greater)
    arr_index_minima = sp.argrelextrema(np.array(dataset_close_val), np.less)
    arr_values_maxima = dataset_close_val.iloc[arr_index_maxima[0]]['Close'].values
    arr_values_minima = dataset_close_val.iloc[arr_index_minima[0]]['Close'].values

    detect_double_formation(arr_index_minima[0], arr_values_minima, 1, chart_data)

def detect_double_formation(arr_index_extreme_values, arr_vals_extreme_values,
type_of_double_formation, chart_data):
    """Detects double bottom/top in chart_data. """

    # 0 = Double Top; 1 = Double Bottom; 2 = Both

    dataset_close_val = chart_data[['Close']]
    found_formations = []
    found_formations_index = []

    crit_compare_extreme_vals = '>='
    neckline_operator = "<"
    title = "Top"
    if type_of_double_formation == 1:
        crit_compare_extreme_vals = '<='
        neckline_operator = ">"
        title = "Bottom"

    for index_arr, index_dataset in enumerate(arr_index_extreme_values): # KANN WEG
        if index_arr > 0:
            prev_extreme = arr_vals_extreme_values[index_arr-1]
            curr_extreme = arr_vals_extreme_values[index_arr]

            if (eval(str(prev_extreme) + crit_compare_extreme_vals + str(curr_extreme))
            and math.isclose(prev_extreme, curr_extreme, rel_tol=0.02)):

                condition_range_between_two_extremes = ((chart_data.index.values >=
                arr_index_extreme_values[index_arr-1]) &
                (chart_data.index.values < arr_index_extreme_values[index_arr]+5))

                arr_range_extremes = chart_data.index.values[condition_range_between_two_extremes]
                arr_values_of_extremes = dataset_close_val[condition_range_between_two_extremes]

                first_index_breaking_neckline = get_first_index_breaking_neckline(
                arr_range_extremes, arr_values_of_extremes['Close'].values, neckline_operator)

                if first_index_breaking_neckline > -1:
                    index_breakthrough = (arr_index_extreme_values[index_arr] +
                    first_index_breaking_neckline)
                    value_breakthrough = dataset_close_val[chart_data.index.values ==
                    index_breakthrough]['Close'].values
                    found_breaking_of_necklines.append([index_breakthrough, value_breakthrough])
                    found_formations.append(prev_extreme)
                    found_formations_index.append(arr_index_extreme_values[index_arr-1])
                    found_formations.append(curr_extreme)
                    found_formations_index.append(arr_index_extreme_values[index_arr])
                    print("Detected Double", title, ":")
                    print(arr_values_of_extremes.index.values)

    # zeichnen --> eigene Methode
    plot_formations(found_formations, found_formations_index, dataset_close_val)

def get_first_index_breaking_neckline(range_arr, value_arr, operator):
    """Gets the first index, which breaks the neckline."""

    start_neckline = min(range_arr)
    end_neckline = max(range_arr)

    if operator == "<":
        neckline_value = np.min(value_arr[:-4])
        outer_condition = all(val < neckline_value for val in value_arr[-4:])
    elif operator == ">":
        neckline_value = np.max(value_arr[:-4])
        outer_condition = all(val > neckline_value for val in value_arr[-4:])

    if not outer_condition:

        for index, val in enumerate(value_arr[-4:]):
            if operator == "<":
                if val < neckline_value:
                    found_necklines.append([neckline_value, start_neckline, end_neckline])
                    return index+1
            elif operator == ">":
                if val > neckline_value:
                    found_necklines.append([neckline_value, start_neckline, end_neckline])
                    return index+1

    return -1


def plot_formations(found_formations, found_formations_index, dataset):
    """Plots chart_data and detected formations."""

    dataset_index = dataset.index.tolist()
    fig_plot = plt.figure(figsize=(80, 40), dpi= 120, facecolor='w', edgecolor='k')
    plt.xticks(range(0, dataset.size))
    plt.plot(found_formations_index, found_formations, 'o', markersize=9.5, color='green')
    plt.plot(dataset_index, dataset, '-', markersize=1.5, color='black', alpha=0.6)

    for i, neckline in enumerate(found_necklines):
        plt.axhline(y=neckline[0], xmin=neckline[1]*(1/dataset.size),
        xmax=neckline[2]*(1/dataset.size), color='red', alpha=0.1)
        plt.plot(found_breaking_of_necklines[i][0], found_breaking_of_necklines[i][1], 'o',
        alpha=1, color='blue')

    plt.xlim([0, dataset.size])
    fig_plot.savefig("plot_Formations.png")

main()
