"""Detects formations in nasdaq-charts. Current: DoubleTop and DoubleBottom."""
from argparse import ArgumentParser, Action
from math import isclose
import time
from pandas import read_csv
import numpy as np
from scipy.signal import argrelextrema
from dateutil.parser import parse
import matplotlib.pyplot as plt

found_necklines = []
found_breaking_of_necklines = []
found_price_targets = []
neckline_value = 0

successful_trades = 0

start = time.time()

class DateParser(Action):
    """Parsing param date to right format."""
    def __call__(self, parser, namespace, values, option_strings=None):
        setattr(namespace, self.dest, parse(values).date().strftime("%Y-%m-%d"))

def main():
    """Set-up argparser and process arguments."""

    # set input parameter and help message
    my_parser = ArgumentParser(description="Detection of Double Top/Bottom formations")
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
        "-f",
        metavar="f",
        default="0",
        help="Set the formations you want to be detected. 0 = Double Top, 1 = Double Bottom")

    my_parser.add_argument(
        "-s",
        metavar="s",
        action=DateParser,
        help="Set the start date of chart data"
    )

    my_parser.add_argument(
        "-e",
        metavar="e",
        action=DateParser,
        help="Set the end date of chart data"
    )

    my_parser.add_argument(
        "-a",
        metavar="a",
        choices=["adjusted", "unadjusted", "splitadjusted"],
        default="unadjusted",
        help="Chart data adjusted or unadjusted"
    )

    args = my_parser.parse_args()

    company, time_frame, formation = args.c.upper(), args.t.lower(), args.f
    start_date, end_date, adjusted = args.s, args.e, args.a

    chart_data = read_file(company, time_frame, adjusted)
    prepare_data(chart_data, formation, company, start_date, end_date)

def read_file(company, time_frame, adjusted):
    """Reads file with chart_data."""

    if time_frame == "day":
        column_names = ["Date","Open", "High","Low","Close","Umsatz (St)",
        "Quelle", "Import-Datum"]
    elif time_frame == "minute":
        column_names = ["Date","Time","Open", "High","Low","Close","Umsatz (St)",
        "Quelle", "Import-Datum"]

    filepath = (fr'/data/financedata/2020ss/gelling/data/kibot3/NASDAQ/{company}/{company.lower()}'
        fr'.candle.{time_frame}.{adjusted}')
    chart_data = read_csv(
        filepath,
        ",",
        header=None,
        names=column_names,
        parse_dates=[0])
    return chart_data

def prepare_data(chart_data, formation, company, start_date, end_date):
    """Prepares data for further calculations. """

    chart_data = chart_data[:50000]

    if start_date:
        if start_date <= max(chart_data.Date).date().strftime("%Y-%m-%d"):
            chart_data = chart_data[chart_data.Date >= start_date]
        else:
            raise Exception("""Start Date higher than dataset dates.
            Please insert an lower start date.""")
    if end_date:
        if end_date >= min(chart_data.Date).date().strftime("%Y-%m-%d"):
            chart_data = chart_data[chart_data.Date <= end_date]
        else:
            raise Exception("""End date lower than dataset dates.
            Please insert an higher end date.""")

    detect_double_formation(formation, chart_data, company)

def detect_double_formation(type_of_double_formation, chart_data, company):
    """Detects double bottom/top in chart_data. """

    #global successful_trades

    start_money = 1000
    # 0 = Double Top; 1 = Double Bottom; 2 = Both
    dataset_close_val = chart_data[['Close']]
    arr_index_extreme_values, arr_vals_extreme_values = [], []
    found_formations_index, found_formations = [], []

    if type_of_double_formation == "0":
        crit_compare_extreme_vals, neckline_operator = '>=', "<"
        arr_index_extreme_values = argrelextrema(np.array(dataset_close_val), np.greater)[0]
        arr_vals_extreme_values=dataset_close_val.iloc[arr_index_extreme_values]['Close'].values

    elif type_of_double_formation == "1":
        crit_compare_extreme_vals, neckline_operator = '<=', ">"
        arr_index_extreme_values = argrelextrema(np.array(dataset_close_val), np.less)[0]
        arr_vals_extreme_values=dataset_close_val.iloc[arr_index_extreme_values]['Close'].values

    for index_arr, index_dataset in enumerate(arr_index_extreme_values):
        if index_arr > 0:
            prev_extreme = arr_vals_extreme_values[index_arr-1]
            curr_extreme = arr_vals_extreme_values[index_arr]

            if (eval(str(prev_extreme) + crit_compare_extreme_vals + str(curr_extreme))
            and isclose(prev_extreme, curr_extreme, rel_tol=0.005)):

                condition_range_between_two_extremes = ((chart_data.index.values >=
                arr_index_extreme_values[index_arr-1]) &
                (chart_data.index.values <= arr_index_extreme_values[index_arr]+9))

                condition_range_between_two_extremes2 = ((chart_data.index.values >=
                arr_index_extreme_values[index_arr-1]) &
                (chart_data.index.values <= arr_index_extreme_values[index_arr]))

                condition_range_between_two_extremes3 = ((chart_data.index.values >
                arr_index_extreme_values[index_arr]) &
                (chart_data.index.values <= arr_index_extreme_values[index_arr]+9))

                arr_range_extremes = chart_data.index.values[condition_range_between_two_extremes]
                arr_values_of_extremes = dataset_close_val[condition_range_between_two_extremes]
                arr_values_between_extremes=dataset_close_val[condition_range_between_two_extremes2]
                arr_values_after_extremes = dataset_close_val[condition_range_between_two_extremes3]

                if type_of_double_formation == "0":
                    cond = np.all(arr_values_of_extremes['Close'].values <= prev_extreme)
                else:
                    cond = np.all(arr_values_of_extremes['Close'].values >= prev_extreme)

                if cond:
                    first_index_breaking_neckline = get_first_index_breaking_neckline(
                    arr_range_extremes, arr_values_between_extremes['Close'].values,
                    arr_values_after_extremes['Close'].values, neckline_operator)
                    if first_index_breaking_neckline > -1:
                        index_breakthrough = (arr_index_extreme_values[index_arr] +
                        first_index_breaking_neckline)
                        value_breakthrough = dataset_close_val[chart_data.index.values ==
                        index_breakthrough]['Close'].values
                        found_breaking_of_necklines.append([index_breakthrough, value_breakthrough])
                        found_formations.append(prev_extreme)
                        found_formations_index.append(arr_index_extreme_values[index_arr-1])
                        found_formations.append(curr_extreme)
                        found_formations_index.append(index_dataset)

                        price_target = calc_price_target(curr_extreme, neckline_operator)
                        stop_loss = calc_stop_loss(neckline_operator)

                        if type_of_double_formation == "0":
                            if np.all(arr_values_after_extremes['Close'].values < stop_loss):
                                start_money = (start_money +
                                (value_breakthrough-arr_values_after_extremes['Close'].values[-1]))
                            else:
                                start_money = start_money + (value_breakthrough - stop_loss)
                        else:
                            if np.all(arr_values_after_extremes['Close'].values > stop_loss):
                                start_money = (start_money +
                                (arr_values_after_extremes['Close'].values[-1]-value_breakthrough))
                            else:
                                start_money = start_money + (stop_loss - value_breakthrough)

                        print("Take profits at Price Target of", price_target)
                        found_price_targets.append([price_target, min(arr_range_extremes),
                        max(arr_range_extremes)])
                        is_successful_trade(arr_values_after_extremes['Close'].values,
                        price_target, neckline_operator)
                        print("Set stop loss at:", stop_loss)
                        print("Indizes:", arr_values_of_extremes.index.values)

    # zeichnen --> eigene Methode
    #f = open("20220714_0K5_9_money_both_changes.txt", "a")
    #f.write(f'\n {company}, {start_money[0]}')
    #plot_formations(found_formations, found_formations_index, dataset_close_val, company)
    end = time.time()
    print("Duration before print:", end - start)
    print(start_money[0])

def is_successful_trade(arr_values_after_extremes, price_target, operator):
    """calculate if trade would have been successful."""
    global successful_trades

    if operator == "<":
        succesful_trade_condition = arr_values_after_extremes <= price_target
    else:
        succesful_trade_condition = arr_values_after_extremes >= price_target

    if sum(np.squeeze(succesful_trade_condition))>0:
        successful_trades += 1

def all_vals_smallerthan_neckline_value(arr_values, neckline_value, operator):
    """Calculate if all values until this point are lower than pricetarget."""

    signs = np.sign(arr_values - neckline_value)
    diff = np.diff(signs[signs != 0])

    if operator == "<":
        return not np.any(diff == 2) #np.all((arr_values <= neckline_value) == True)

    return not np.any(diff == -2) #np.all((arr_values >= neckline_value) == True)

def calc_price_target(snd_extrempoint, operator):
    """calculate price target of formation."""

    if operator == "<":
        return neckline_value - (snd_extrempoint - neckline_value)

    return neckline_value + (neckline_value - snd_extrempoint)

def calc_stop_loss(operator):
    """Calculate where to set the stop loss."""

    if operator == "<":
        return neckline_value * 1.01

    return neckline_value * 0.99

def get_first_index_breaking_neckline(range_arr, values_between_extremes_arr,
                                      values_after_extremes_arr, operator):
    """Gets the first index, which breaks the neckline."""
    global neckline_value
    start_neckline, end_neckline = min(range_arr), max(range_arr)

    if operator == "<":
        neckline_value = np.min(values_between_extremes_arr)
        outer_condition = all(val < neckline_value for val in values_after_extremes_arr)
    elif operator == ">": # bottom
        neckline_value = np.max(values_between_extremes_arr)
        outer_condition = all(val > neckline_value for val in values_after_extremes_arr)

    if not outer_condition:
        for index, val in enumerate(values_after_extremes_arr):
            if operator == "<":
                if val < neckline_value:
                    found_necklines.append([neckline_value, start_neckline, end_neckline])
                    print("\nDETECTED DOUBLE TOP")
                    print("Short gehen und")
                    return index+1
            elif operator == ">":
                if val > neckline_value:
                    found_necklines.append([neckline_value, start_neckline, end_neckline])
                    print("\nDETECTED DOUBLE BOTTOM")
                    print("Long gehen und")
                    return index+1

    return -1

def plot_formations(found_formations, found_formations_index, dataset, company):
    """Plots chart_data and detected formations."""

    dataset_index = dataset.index.tolist() # 320, 80
    fig_plot = plt.figure(figsize=(320, 80), dpi= 100, facecolor='w', edgecolor='k')
    #ax = fig_plot.add_subplot(111)
    plt.xticks(range(0, dataset.size))
    plt.plot(found_formations_index, found_formations, 'o', markersize=9.5, color='green')
    plt.plot(dataset_index, dataset, '-', markersize=1.5, color='black', alpha=0.6)

    #for i,j in zip(found_formations_index,found_formations):
        #ax.annotate(str(j),xy=(i,j)) # Beschriftung von Extrema
    for i, neckline in enumerate(found_necklines):
        plt.axhline(y=neckline[0], xmin=neckline[1]*(1/dataset.size),
        xmax=neckline[2]*(1/dataset.size), color='red', alpha=0.1)
        plt.plot(found_breaking_of_necklines[i][0], found_breaking_of_necklines[i][1], 'o',
        alpha=1, color='blue')
        plt.axhline(y=found_price_targets[i][0], xmin=found_price_targets[i][1]*(1/dataset.size),
        xmax=(found_price_targets[i][2]-1)*(1/dataset.size), color='yellow', alpha=0.1)

    plt.xlim([0, dataset.size])
    fig_plot.savefig(fr'plot_formations2_{company}.png')
    end = time.time()
    print("Duration:", end - start)
main()
