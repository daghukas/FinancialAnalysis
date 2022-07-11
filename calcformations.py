"""Detects formations in nasdaq-charts. Current: DoubleTop and DoubleBottom."""
from argparse import ArgumentParser, Action
from math import isclose
from pandas import read_csv
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import time
#import datetime
from dateutil.parser import parse, ParserError

found_necklines = []
found_breaking_of_necklines = []
found_price_targets = []
neckline_value = 0

successful_trades = 0
successful_trades_first_index = 0
successful_trades_second_index = 0
successful_trades_third_index = 0
successful_trades_fourth_index = 0
successful_trades_fifth_index = 0
successful_trades_sixth_index = 0
successful_trades_seventh_index = 0
successful_trades_eigth_index = 0
successful_trades_ninth_index = 0
successful_trades_tenth_index = 0
successful_trades_eleventh_index = 0
successful_trades_twelveth_index = 0
successful_trades_thirteenth_index = 0

start = time.time()

class DateParser(Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        setattr(namespace, self.dest, parse(values).date().strftime("%Y-%m-%d"))

def main():
    """Set-up argparser and process arguments."""

    # set input parameter and help message
    my_parser = ArgumentParser()
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

    args = my_parser.parse_args()

    company, time_frame, formation, start_date, end_date = args.c.upper(), args.t.lower(), args.f, args.s, args.e
    print(start_date)
    print(end_date)
    chart_data = read_file(company, time_frame)
    prepare_data(chart_data, formation, company, start_date, end_date)

def read_file(company, time_frame):
    """Reads file with chart_data."""

    if time_frame == "day":
        column_names = ["Date","Open", "High","Low","Close","Umsatz (St)", "Quelle", "Import-Datum"]
    elif time_frame == "minute":
        column_names = ["Date","Time","Open", "High","Low","Close","Umsatz (St)", "Quelle", "Import-Datum"]

    filepath = (fr'/data/financedata/2020ss/gelling/data/kibot3/NASDAQ/{company}/{company.lower()}'
        fr'.candle.{time_frame}.unadjusted')
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
            raise Exception("Start Date higher than dataset dates. Please insert an lower start date.") 
    if end_date:
        if end_date >= min(chart_data.Date).date().strftime("%Y-%m-%d"):
            chart_data = chart_data[chart_data.Date <= end_date]
        else:
            raise Exception("End date lower than dataset dates. Please insert an higher end date.")

    detect_double_formation(formation, chart_data, company)

def detect_double_formation(type_of_double_formation, chart_data, company):
    """Detects double bottom/top in chart_data. """

    global successful_trades
    global successful_trades_first_index
    global successful_trades_second_index
    global successful_trades_third_index
    global successful_trades_fourth_index
    global successful_trades_fifth_index
    global successful_trades_sixth_index
    global successful_trades_seventh_index
    global successful_trades_eigth_index
    global successful_trades_ninth_index
    global successful_trades_tenth_index
    global successful_trades_eleventh_index
    global successful_trades_twelveth_index
    global successful_trades_thirteenth_index

    # 0 = Double Top; 1 = Double Bottom; 2 = Both
    dataset_close_val = chart_data[['Close']]
    arr_index_extreme_values, arr_vals_extreme_values = [], []
    found_formations_index, found_formations = [], []

    if type_of_double_formation == "0":
        crit_compare_extreme_vals, neckline_operator = '>=', "<"
        arr_index_extreme_values = argrelextrema(np.array(dataset_close_val), np.greater)[0]
        arr_vals_extreme_values = dataset_close_val.iloc[arr_index_extreme_values]['Close'].values

    elif type_of_double_formation == "1":
        crit_compare_extreme_vals, neckline_operator = '<=', ">"
        arr_index_extreme_values = argrelextrema(np.array(dataset_close_val), np.less)[0]
        arr_vals_extreme_values = dataset_close_val.iloc[arr_index_extreme_values]['Close'].values

    for index_arr, index_dataset in enumerate(arr_index_extreme_values):
        if index_arr > 0:
            prev_extreme = arr_vals_extreme_values[index_arr-1]
            curr_extreme = arr_vals_extreme_values[index_arr]

            if (eval(str(prev_extreme) + crit_compare_extreme_vals + str(curr_extreme))
            and isclose(prev_extreme, curr_extreme, rel_tol=0.005)):

                condition_range_between_two_extremes = ((chart_data.index.values >=
                arr_index_extreme_values[index_arr-1]) &
                (chart_data.index.values <= arr_index_extreme_values[index_arr]+9)) #vorher 4

                condition_range_between_two_extremes2 = ((chart_data.index.values >=
                arr_index_extreme_values[index_arr-1]) &
                (chart_data.index.values <= arr_index_extreme_values[index_arr]))

                condition_range_between_two_extremes3 = ((chart_data.index.values >
                arr_index_extreme_values[index_arr]) &
                (chart_data.index.values <= arr_index_extreme_values[index_arr]+9)) #vorher 4

                arr_range_extremes = chart_data.index.values[condition_range_between_two_extremes]
                arr_values_of_extremes = dataset_close_val[condition_range_between_two_extremes]
                arr_values_between_extremes = dataset_close_val[condition_range_between_two_extremes2]
                arr_values_after_extremes = dataset_close_val[condition_range_between_two_extremes3]

                first_index_breaking_neckline = get_first_index_breaking_neckline(
                arr_range_extremes, arr_values_between_extremes['Close'].values, arr_values_after_extremes['Close'].values, neckline_operator)
                
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
                    #print("Take profits at Price Target of",price_target)
                    found_price_targets.append([price_target, min(arr_range_extremes),
                    max(arr_range_extremes)])
                    #print("Whole Range", arr_range_extremes)
                    #print("Between Extremes", arr_values_between_extremes)
                    #print("After Extremes", arr_values_after_extremes, "PT", price_target)
                    is_successful_trade(arr_values_after_extremes['Close'].values, arr_values_between_extremes['Close'].values, price_target, neckline_operator)
                    print("Set stop loss at:", stop_loss)
                    print("Indizes:", arr_values_of_extremes.index.values)
                    #print("Values:", arr_values_of_extremes['Close'].values) war vorher weg

    # zeichnen --> eigene Methode
    #f = open("20220704_BT_2_num_detected_formation.txt", "a")
    #f.write(f'\n {company}, {len(found_breaking_of_necklines)}')
    end = time.time()
    print("Duration before print:", end - start)

    #f = open("BTnum_of_detected_forms0K5Prozent.txt", "a")
    #f.write(f'\n {company}, {len(found_breaking_of_necklines)}, {successful_trades}, {successful_trades_first_index}, {successful_trades_second_index}, {successful_trades_third_index}, {successful_trades_fourth_index}, {successful_trades_fifth_index}, {successful_trades_sixth_index}, {successful_trades_seventh_index}, {successful_trades_eigth_index}, {successful_trades_ninth_index}, {successful_trades_tenth_index}, {successful_trades_eleventh_index}, {successful_trades_twelveth_index}, {successful_trades_thirteenth_index}')
    # plot_formations(found_formations, found_formations_index, dataset_close_val, company)
    #print(len(found_breaking_of_necklines))
    #print(successful_trades)
    #print(successful_trades_first_index, successful_trades_second_index, successful_trades_third_index,
    #successful_trades_fourth_index, successful_trades_fifth_index, successful_trades_sixth_index, 
    #successful_trades_seventh_index, successful_trades_eigth_index, successful_trades_ninth_index,
    #successful_trades_tenth_index, successful_trades_eleventh_index, successful_trades_twelveth_index,
    #successful_trades_thirteenth_index)

def is_successful_trade(arr_values_after_extremes, values_between_extremes_arr, price_target, operator):
    """calculate if trade would have been successful."""
    global successful_trades
    global successful_trades_first_index
    global successful_trades_second_index
    global successful_trades_third_index
    global successful_trades_fourth_index
    global successful_trades_fifth_index
    global successful_trades_sixth_index
    global successful_trades_seventh_index
    global successful_trades_eigth_index
    global successful_trades_ninth_index
    global successful_trades_tenth_index
    global successful_trades_eleventh_index
    global successful_trades_twelveth_index
    global successful_trades_thirteenth_index

    if operator == "<":
        neckline_value = np.min(values_between_extremes_arr)
        succesful_trade_condition = arr_values_after_extremes <= price_target
        compare_value_with_price_target = "<="
    else:
        neckline_value = np.max(values_between_extremes_arr)
        succesful_trade_condition = arr_values_after_extremes >= price_target
        compare_value_with_price_target = ">=" 

    if sum(np.squeeze(succesful_trade_condition))>0:
        successful_trades += 1

    # darf nicht unter und dann wieder uber NL gehen, darf aber uber NL sein
    for index, val in enumerate(arr_values_after_extremes):
        if (eval(str(val) + compare_value_with_price_target + str(price_target))):
            if index == 0:
                successful_trades_first_index += 1
                #print("BEI", index, arr_values_after_extremes)
            elif index == 1:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_second_index += 1
                    #print("BEI", index, arr_values_after_extremes)
            elif index == 2:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_third_index += 1
                    #print("BEI", index, arr_values_after_extremes)
            elif index == 3:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_fourth_index += 1
                    #print("BEI", index, arr_values_after_extremes)
            elif index == 4:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_fifth_index += 1  #successful_trades_sixth_index
            elif index == 5:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_sixth_index += 1
            elif index == 6:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_seventh_index += 1
            elif index == 7:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_eigth_index += 1
            elif index == 8:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_ninth_index += 1
            elif index == 9:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_tenth_index += 1
            elif index == 10:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_eleventh_index += 1
            elif index == 11:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_twelveth_index += 1
            elif index == 12:
                if all_vals_smallerthan_neckline_value(arr_values_after_extremes[:index+1], neckline_value, operator):
                    successful_trades_thirteenth_index += 1
            else:
                None
    
def all_vals_smallerthan_neckline_value(arr_values, neckline_value, operator):
    """Calculate if all values until this point are lower than pricetarget."""
    # nicht alle Vals, sondern wenn man einmal unter der Nackenlinie ist, darf man nicht mehr druber
    # VERBESSERN

    signs = np.sign(arr_values - neckline_value)
    diff = np.diff(signs[signs != 0])
   
    if operator == "<":
        return not np.any(diff == 2) #np.all((arr_values <= neckline_value) == True)
    else:
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

def get_first_index_breaking_neckline(range_arr, values_between_extremes_arr, values_after_extremes_arr, operator): #value_arr
    """Gets the first index, which breaks the neckline."""
    global neckline_value
    start_neckline, end_neckline = min(range_arr), max(range_arr)
    # top
    #print(neckline_value)
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
    fig_plot.savefig(fr'Plots/plot_formations_{company}.png')
    end = time.time()
    print("Duration:", end - start)
main()
