"""Detects formations in nasdaq-charts. Current: DoubleTop and DoubleBottom."""
from argparse import ArgumentParser, Action
from math import isclose, ceil
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

    chart_data = chart_data[:1000]

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

    global neckline_value
    counter = 0
    min_length = 5

    start_money = 1000
    # 0 = Double Top; 1 = Double Bottom; 2 = Both
    dataset_close_val = chart_data[['Close']]
    arr_index_extreme_values, arr_vals_extreme_values = [], []
    found_formations_index, found_formations = [], []

    if type_of_double_formation == "0":
        crit_compare_extreme_vals, neckline_operator = '>=', "<"
        #arr_index_of_extreme_values = argrelextrema(np.array(dataset_close_val), np.greater)[0]
        arr_index_of_extreme_values = (np.diff(np.sign(np.diff(dataset_close_val['Close']))) < 0).nonzero()[0] + 1
        arr_vals_of_extreme_values=dataset_close_val.iloc[arr_index_of_extreme_values]['Close'].values

    elif type_of_double_formation == "1":
        crit_compare_extreme_vals, neckline_operator = '<=', ">"
        #arr_index_of_extreme_values = argrelextrema(np.array(dataset_close_val), np.less)[0]
        arr_index_of_extreme_values = (np.diff(np.sign(np.diff(dataset_close_val['Close']))) > 0).nonzero()[0] + 1
        arr_vals_of_extreme_values=dataset_close_val.iloc[arr_index_of_extreme_values]['Close'].values

    # index im array aller extrema, index im datenset
    # alle Extremepunkte durchlaufen
    for index_arr, index_dataset in enumerate(arr_index_of_extreme_values):

        curr_extreme = arr_vals_of_extreme_values[index_arr]
        # Extrempunkte die mind. 30 Einheiten und max. 180 Einheiten entfernt sind (1M - 3M)
        following_extremes_indexes_filtered = arr_index_of_extreme_values[(arr_index_of_extreme_values >= (index_dataset + min_length)) & (arr_index_of_extreme_values <= (index_dataset + 180))]
        
        # entfernte Extrempunkte durchlaufen
        for index_of_filtered_extreme_value in following_extremes_indexes_filtered:

            following_extreme_value = dataset_close_val.iloc[chart_data.index.values == index_of_filtered_extreme_value]['Close'].values[0]

            # entfernte EP und aktueller EP Differenz-Check
            if (eval(str(curr_extreme) + crit_compare_extreme_vals + str(following_extreme_value))
            and isclose(curr_extreme, following_extreme_value, rel_tol=0.005)):

                condition_range_between_two_extremes = ((chart_data.index.values >=
                arr_index_of_extreme_values[index_arr]) &
                (chart_data.index.values <= index_of_filtered_extreme_value))
                
                # Werte zwischen Extrempunkte
                arr_values_between_extremes = dataset_close_val[condition_range_between_two_extremes]
                len_of_formation = len(arr_values_between_extremes[1:-1])

                condition_complete_range = ((chart_data.index.values >=
                arr_index_of_extreme_values[index_arr]) &
                (chart_data.index.values <= (index_of_filtered_extreme_value+len_of_formation)))

                condition_range_after_snd_extreme = ((chart_data.index.values >
                index_of_filtered_extreme_value) &
                (chart_data.index.values <= (index_of_filtered_extreme_value+len_of_formation)))

                # Komplette Range mit nach 2. Extrempunkt
                arr_index_complete_range = chart_data.index.values[condition_complete_range]
                arr_values_complete_range = dataset_close_val[condition_complete_range]

                # Werte nach zweiten Extrempunkt
                arr_values_after_extremes = dataset_close_val[condition_range_after_snd_extreme]

                if type_of_double_formation == "0":
                    neckline_value = np.min(arr_values_between_extremes['Close'].values)

                    must_true = (np.all(arr_values_between_extremes[1:-2]['Close'].values < curr_extreme) & 
                    np.all(arr_values_between_extremes[4:-2]['Close'].values < following_extreme_value) &
                    np.all(arr_values_after_extremes['Close'].values < following_extreme_value))
                else:
                    neckline_value = np.max(arr_values_between_extremes['Close'].values)

                    must_true = (np.all(arr_values_between_extremes[1:-2]['Close'].values > curr_extreme) &
                    np.all(arr_values_between_extremes[4:-2]['Close'].values > following_extreme_value) &
                    np.all(arr_values_after_extremes['Close'].values > following_extreme_value))

                # CHECKEN, DASS ZWISCHENDRIN NICHT MEHR AUF ERSTES TOP ZRK KOMME; SONDERN ABSTAND MEHR ALS 0;5 PROZENT ZUM TOP IST
                # statt 4, vllt berechnen wenn > min_length dann 4, sonder min_length/2
                #must_true

                # vllt laenge durch 3 anfangs ausschliessen math.ceil(4.2)
                if ((curr_extreme*0.99499 >= neckline_value) & must_true):
                    # cond implementieren und weitermachen
                    #if type_of_double_formation == "0":
                    #    condition_values_after_extreme = np.all(arr_values_complete_range['Close'].values <= curr_extreme)
                    #else:
                    #    condition_values_after_extreme = np.all(arr_values_complete_range['Close'].values >= curr_extreme)

                    #if condition_values_after_extreme & check_previous_trend(chart_data, index_arr, neckline_value, type_of_double_formation, len_of_formation):
                    if check_previous_trend(chart_data, index_arr, neckline_value, type_of_double_formation, len_of_formation):

                        first_index_breaking_neckline = get_first_index_breaking_neckline(
                        arr_index_complete_range, arr_values_between_extremes['Close'].values,
                        arr_values_after_extremes['Close'].values, neckline_operator)

                        if first_index_breaking_neckline > -1:
                            print("Complete", arr_values_complete_range)
                            print("1st Extreme", index_dataset)
                            print("2snd Extreme", index_of_filtered_extreme_value)
                            print("Snd extreme:", following_extreme_value)
                            print("After 2snd", arr_values_after_extremes['Close'].values)
                            print("NecklineVal", neckline_value)
                            print("Durchbruch!!!")
                            index_breakthrough = (index_of_filtered_extreme_value +
                            first_index_breaking_neckline + 1)
                            
                            value_breakthrough = dataset_close_val[chart_data.index.values ==
                            index_breakthrough]['Close'].values
                            print("I-B", index_breakthrough)
                            print("V-B", value_breakthrough)
                            found_breaking_of_necklines.append([index_breakthrough, value_breakthrough])
                            found_formations.append(curr_extreme)
                            found_formations_index.append(index_dataset)
                            found_formations.append(following_extreme_value)
                            found_formations_index.append(index_of_filtered_extreme_value)

                            price_target = calc_price_target(following_extreme_value, neckline_operator)
                            stop_loss = calc_stop_loss(neckline_operator)
                            counter += 1
                            #print("PT", price_target, "SL", stop_loss)
                            found_price_targets.append([price_target, min(arr_index_complete_range),
                            max(arr_index_complete_range)])

    print("Anzahl", counter)
    plot_formations(found_formations, found_formations_index, dataset_close_val, company, arr_index_of_extreme_values, arr_vals_of_extreme_values)

def check_previous_trend(chart_data, index_first_extreme, neckline_value, type_of_double_formation, len_of_formation):
    """Check trend before first extreme, DT=from down, BT=from up"""

    # nicht 9 sondern an langer zw Extrema anpassen
    previous_data = chart_data[['Close']][((chart_data.index.values < index_first_extreme) & 
    (chart_data.index.values >= (index_first_extreme - ceil(len_of_formation*1.5))))]['Close'].values

    signs = np.sign(previous_data - neckline_value)
    diff = np.diff(signs[signs != 0])

    if type_of_double_formation == "0":
        if len(diff[diff==2]) > 0:
            print("Previous Data:", previous_data)
            print("Neckline Val:", neckline_value)
            print("Signs:", signs)
            print("Diff of Signs:", diff[diff==2])
            return True
    else:
        if len(diff[diff==-2]) > 0:
            print("Previous Data:", previous_data)
            print("Neckline Val:", neckline_value)
            print("Signs:", signs)
            print("Diff of Signs:", diff[diff==-2])
            return True

    return False

def get_first_index_breaking_neckline(arr_index_complete_range, values_between_extremes_arr,
                                      values_after_extremes_arr, operator):
    """Gets the first index, which breaks the neckline."""
    global neckline_value
    start_neckline, end_neckline = min(arr_index_complete_range), max(arr_index_complete_range)

    if operator == "<":
        outer_condition = all(val < neckline_value for val in values_after_extremes_arr)
    elif operator == ">": # bottom
        outer_condition = all(val > neckline_value for val in values_after_extremes_arr)

    #if not outer_condition:
    for index, val in enumerate(values_after_extremes_arr):
        if operator == "<":
            if val < neckline_value:
                found_necklines.append([neckline_value, start_neckline, end_neckline])
                print("\nDETECTED DOUBLE TOP")
                print("Short gehen und")
                print("BEI INDEX", index)
                return index
        else:
            if val > neckline_value:
                found_necklines.append([neckline_value, start_neckline, end_neckline])
                print("\nDETECTED DOUBLE BOTTOM")
                print("Long gehen und")
                return index

    return -1

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

def plot_formations(found_formations, found_formations_index, dataset, company, arr_index_of_extreme_values, arr_vals_of_extreme_values):
    """Plots chart_data and detected formations."""
    dataset_index = dataset.index.tolist() # 320, 80
    fig_plot = plt.figure(figsize=(320, 80), dpi= 100, facecolor='w', edgecolor='k')
    #ax = fig_plot.add_subplot(111)
    plt.xticks(range(0, dataset.size))
    plt.plot(found_formations_index, found_formations, 'o', markersize=9.5, color='green')
    #plt.plot(arr_index_of_extreme_values, arr_vals_of_extreme_values, 'o', markersize=9.5, color='green')
    plt.plot(dataset_index, dataset, '-', markersize=4, color='black', alpha=0.6)

    for i, neckline in enumerate(found_necklines):
        plt.axhline(y=neckline[0], xmin=neckline[1]*(1/dataset.size),
        xmax=neckline[2]*(1/dataset.size), color='red', alpha=0.1)
        plt.plot(found_breaking_of_necklines[i][0], found_breaking_of_necklines[i][1], 'o',
        alpha=1, color='blue')
        plt.axhline(y=found_price_targets[i][0], xmin=found_price_targets[i][1]*(1/dataset.size),
        xmax=(found_price_targets[i][2]-1)*(1/dataset.size), color='yellow', alpha=0.1)

    plt.xlim([0, dataset.size])
    fig_plot.savefig(fr'plot_formations_run_{company}.png')

main()