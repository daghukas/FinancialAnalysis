"""Detects formations in nasdaq-charts. Current: DoubleTop and DoubleBottom."""
from argparse import ArgumentParser, Action
from math import isclose, ceil
import time
from pandas import read_csv
import numpy as np
from scipy.signal import argrelextrema
from dateutil.parser import parse
import matplotlib.pyplot as plt
from collections import Counter

found_necklines = []
found_breaking_of_necklines = []
found_price_targets = []
successful_trade = []
successful_trade_0 = []
successful_trade_1 = []
successful_trade_2 = []
successful_trade_3 = []
successful_trade_4 = []
successful_trade_5 = []
successful_trade_6 = []
successful_trade_7 = []
successful_trade_8 = []
successful_trade_9 = []
successful_trade_10 = []
successful_trade_11 = []
successful_trade_12 = []
successful_trade_13 = []
successful_trade_14 = []
successful_trade_15 = []
successful_trade_16 = []
successful_trade_17 = []


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
    global successful_trade
    global successful_trade_0
    global successful_trade_1
    global successful_trade_2
    global successful_trade_3
    global successful_trade_4
    global successful_trade_5
    global successful_trade_6
    global successful_trade_7
    global successful_trade_8
    global successful_trade_9
    global successful_trade_10
    global successful_trade_11
    global successful_trade_12
    global successful_trade_13
    global successful_trade_14
    global successful_trade_15
    global successful_trade_16
    global successful_trade_17

    counter = 0
    min_length = 5

    start_money = 1000
    # 0 = Double Top; 1 = Double Bottom; 2 = Both
    dataset_close_val = chart_data[['Close']]
    first_close_val = chart_data[['Close']]['Close'].values[0]
    last_close_val = chart_data[['Close']]['Close'].values[-1]
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
            and isclose(curr_extreme, following_extreme_value, rel_tol=0.01)):

                condition_range_between_two_extremes = ((chart_data.index.values >=
                arr_index_of_extreme_values[index_arr]) &
                (chart_data.index.values <= index_of_filtered_extreme_value))
                
                # Werte zwischen Extrempunkte
                arr_values_between_extremes = dataset_close_val[condition_range_between_two_extremes]
                len_of_formation = len(arr_values_between_extremes[1:-1])

                condition_complete_range = ((chart_data.index.values >=
                arr_index_of_extreme_values[index_arr]) &
                (chart_data.index.values <= (index_of_filtered_extreme_value+(len_of_formation))))

                condition_range_after_snd_extreme = ((chart_data.index.values >
                index_of_filtered_extreme_value) &
                (chart_data.index.values <= (index_of_filtered_extreme_value+(len_of_formation))))

                # Komplette Range mit nach 2. Extrempunkt
                arr_index_complete_range = chart_data.index.values[condition_complete_range]
                arr_values_complete_range = dataset_close_val[condition_complete_range]

                # Werte nach zweiten Extrempunkt
                arr_values_after_extremes = dataset_close_val[condition_range_after_snd_extreme]

                if type_of_double_formation == "0":
                    neckline_value = np.min(arr_values_between_extremes['Close'].values)
                    multiplicator = 0.98999
                    must_true = (np.all(arr_values_between_extremes[1:-2]['Close'].values < curr_extreme) & 
                    np.all(arr_values_between_extremes[4:-2]['Close'].values < following_extreme_value) &
                    np.all(arr_values_after_extremes['Close'].values < following_extreme_value))
                else:
                    neckline_value = np.max(arr_values_between_extremes['Close'].values)
                    multiplicator = 1.01001
                    must_true = (np.all(arr_values_between_extremes[1:-2]['Close'].values > curr_extreme) &
                    np.all(arr_values_between_extremes[4:-2]['Close'].values > following_extreme_value) &
                    np.all(arr_values_after_extremes['Close'].values > following_extreme_value))

                # CHECKEN, DASS ZWISCHENDRIN NICHT MEHR AUF ERSTES TOP ZRK KOMME; SONDERN ABSTAND MEHR ALS 0;5 PROZENT ZUM TOP IST
                # statt 4, vllt berechnen wenn > min_length dann 4, sonder min_length/2

                if check_diff_to_neckline_value(curr_extreme, type_of_double_formation) & must_true:

                    if check_previous_trend(chart_data, index_dataset, neckline_value, type_of_double_formation, len_of_formation):

                        first_index_breaking_neckline = get_first_index_breaking_neckline(
                        arr_index_complete_range, arr_values_between_extremes['Close'].values,
                        arr_values_after_extremes['Close'].values, neckline_operator)

                        if first_index_breaking_neckline > -1:
                            print("Complete Formation", arr_values_complete_range)
                            print("1st Extreme", index_dataset)
                            print("2snd Extreme", index_of_filtered_extreme_value)
                            #print("Snd extreme:", following_extreme_value)
                            #print("After 2snd", arr_values_after_extremes['Close'].values)
                            #print("NecklineVal", neckline_value)
                            #print("Durchbruch!!!")
                            index_breakthrough = (index_of_filtered_extreme_value +
                            first_index_breaking_neckline + 1)
                            
                            value_breakthrough = dataset_close_val[chart_data.index.values ==
                            index_breakthrough]['Close'].values
                            #print("I-B", index_breakthrough)
                            #print("V-B", value_breakthrough)
                            found_breaking_of_necklines.append([index_breakthrough, value_breakthrough])
                            found_formations.append(curr_extreme)
                            found_formations_index.append(index_dataset)
                            found_formations.append(following_extreme_value)
                            found_formations_index.append(index_of_filtered_extreme_value)

                            price_target = calc_price_target(following_extreme_value, neckline_operator)
                            stop_loss = calc_stop_loss(neckline_operator)
                            is_successful_trade(arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values, value_breakthrough, neckline_operator, stop_loss)

                            if type_of_double_formation == "0":
                                # nach Durchbruch-Index
                                #print("Nach Durchbruch-Index:", arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values)
                                if np.all(arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values < stop_loss):
                                    if len(arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values) > 0:
                                        if len(arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values) >= 14:
                                            index = 13
                                        else:
                                            index = -1
                                        #print("Index", index)
                                        #print("Val", arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values[index])
                                        start_money = (start_money +
                                        (value_breakthrough-arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values[index]))# vorher -1 statt 0
                                        #print("WIN")
                                        #print("NEW Money:", start_money[0])
                                else:
                                    start_money = start_money + (value_breakthrough - stop_loss)
                                    #print("LOSE")
                                    #print("NEW Money:", start_money[0])

                            else:
                                if np.all(arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values > stop_loss):
                                    if len(arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values) > 0:
                                        start_money = (start_money +
                                        (arr_values_after_extremes[(index_breakthrough - index_of_filtered_extreme_value):]['Close'].values[-1]-value_breakthrough))
                                        #print("WIN")
                                        #print("NEW Money:", start_money[0])
                                else:
                                    start_money = start_money + (stop_loss - value_breakthrough)
                                    #print("LOSE")
                                    #print("NEW Money:", start_money[0])

                            counter += 1
                            print("SL", stop_loss)
                            found_price_targets.append([price_target, min(arr_index_complete_range),
                            max(arr_index_complete_range)])

    print("Anzahl:", counter)
    print("Money:", start_money[0])
    #f = open("20220818_StopLoss/DT3000DS_1Prozent.txt", "a")
    #if counter == 0:
    #    f.write(f'\n {company}, {counter}, {first_close_val}, {last_close_val}, {start_money}')
    #else:
    #    f.write(f'\n {company}, {counter}, {first_close_val}, {last_close_val}, {start_money[0]}')

    #f.write(f'\n {company}, {counter},')
    #if len(successful_trade_0):
    #    line = f'{len(successful_trade_0)}, {sum(successful_trade_0) / len(successful_trade_0)}, '.replace("\n"," ")
    #    f.write(line)
    #if len(successful_trade_1):
    #    f.write(f'{len(successful_trade_1)}, {sum(successful_trade_1) / len(successful_trade_1)}')
    #if len(successful_trade_2):
    #    f.write(f',{len(successful_trade_2)}, {sum(successful_trade_2) / len(successful_trade_2)}')
    #if len(successful_trade_3):
    #    f.write(f',{len(successful_trade_3)}, {sum(successful_trade_3) / len(successful_trade_3)}')
    #if len(successful_trade_4):
    #    f.write(f',{len(successful_trade_4)}, {sum(successful_trade_4) / len(successful_trade_4)}')
    #if len(successful_trade_5):
    #    f.write(f',{len(successful_trade_5)}, {sum(successful_trade_5) / len(successful_trade_5)}')
    #if len(successful_trade_6):
    #    f.write(f',{len(successful_trade_6)}, {sum(successful_trade_6) / len(successful_trade_6)}')
    #if len(successful_trade_7):
    #    f.write(f',{len(successful_trade_7)}, {sum(successful_trade_7) / len(successful_trade_7)}')
    #if len(successful_trade_8):
    #    f.write(f',{len(successful_trade_8)}, {sum(successful_trade_8) / len(successful_trade_8)}')
    #if len(successful_trade_9):
    #    f.write(f',{len(successful_trade_9)}, {sum(successful_trade_9) / len(successful_trade_9)}')
    #if len(successful_trade_10):
    #    f.write(f',{len(successful_trade_10)}, {sum(successful_trade_10) / len(successful_trade_10)}')
    #if len(successful_trade_11):
    #    f.write(f',{len(successful_trade_11)}, {sum(successful_trade_11) / len(successful_trade_11)}')
    #if len(successful_trade_12):
    #    f.write(f',{len(successful_trade_12)}, {sum(successful_trade_12) / len(successful_trade_12)}')
    #if len(successful_trade_13):
    #    f.write(f',{len(successful_trade_13)}, {sum(successful_trade_13) / len(successful_trade_13)}')
    #if len(successful_trade_14):
    #    f.write(f',{len(successful_trade_14)}, {sum(successful_trade_14) / len(successful_trade_14)}')
    #if len(successful_trade_15):
    #    f.write(f',{len(successful_trade_15)}, {sum(successful_trade_15) / len(successful_trade_15)}')
    #if len(successful_trade_16):
    #    f.write(f',{len(successful_trade_16)}, {sum(successful_trade_16) / len(successful_trade_16)}')
    #if len(successful_trade_17):
    #    f.write(f',{len(successful_trade_17)}, {sum(successful_trade_17) / len(successful_trade_17)}')

    # gucken, wie man das in Excel eintraegt zum Auswerten
    #print(len(successful_trade_0), sum(successful_trade_0) / len(successful_trade_0))
    #print(len(successful_trade_1), sum(successful_trade_1) / len(successful_trade_1))
    #print(len(successful_trade_2), sum(successful_trade_2) / len(successful_trade_2))
    #print(len(successful_trade_3), sum(successful_trade_3) / len(successful_trade_3))
    #print(len(successful_trade_4), sum(successful_trade_4) / len(successful_trade_4))
    #print(len(successful_trade_5), sum(successful_trade_5) / len(successful_trade_5))
    #print(len(successful_trade_6), sum(successful_trade_6) / len(successful_trade_6))
    #print(len(successful_trade_7), sum(successful_trade_7) / len(successful_trade_7))
    #print(len(successful_trade_8), sum(successful_trade_8) / len(successful_trade_8))
    #plot_formations(found_formations, found_formations_index, dataset_close_val, company, arr_index_of_extreme_values, arr_vals_of_extreme_values)

def is_successful_trade(arr_values_after_breakthrough, value_breakthrough, operator, stop_loss):
    """Calculates if trade is successful."""
    global neckline_value
    global successful_trade
    global successful_trade_0
    global successful_trade_1
    global successful_trade_2
    global successful_trade_3
    global successful_trade_4
    global successful_trade_5
    global successful_trade_6
    global successful_trade_7
    global successful_trade_8
    global successful_trade_9
    global successful_trade_10
    global successful_trade_11
    global successful_trade_12
    global successful_trade_13
    global successful_trade_14
    global successful_trade_15
    global successful_trade_16
    global successful_trade_17

    compare_value_with_price_target = "<="

    #print("After BT:", arr_values_after_breakthrough)
    for index, val in enumerate(arr_values_after_breakthrough):
        
        if (np.all(arr_values_after_breakthrough[index:] < stop_loss)):
            if index == 0:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_0.append(profit)
            elif index == 1:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_1.append(profit)
            elif index == 2:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_2.append(profit)
            elif index == 3:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_3.append(profit)
            elif index == 4:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_4.append(profit)
            elif index == 5:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_5.append(profit)
            elif index == 6:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_6.append(profit)
            elif index == 7:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_7.append(profit)
            elif index == 8:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_8.append(profit)
            elif index == 9:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_9.append(profit)
            elif index == 10:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_10.append(profit)
            elif index == 11:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_11.append(profit)
            elif index == 12:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_12.append(profit)
            elif index == 13:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit, "Val", val)
                successful_trade_13.append(profit)
            elif index == 14:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_14.append(profit)
            elif index == 15:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_15.append(profit)
            elif index == 16:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_16.append(profit)
            elif index == 17:
                profit = (value_breakthrough[0] - val)
                #print("Index:", index, "Profit:", profit)
                successful_trade_17.append(profit)

        else:
            if index == 0:
                successful_trade_0.append(value_breakthrough[0] - stop_loss)
            elif index == 1:
                successful_trade_1.append(value_breakthrough[0] - stop_loss)
            elif index == 2:
                successful_trade_2.append(value_breakthrough[0] - stop_loss)
            elif index == 3:
                successful_trade_3.append(value_breakthrough[0] - stop_loss)
            elif index == 4:
                successful_trade_4.append(value_breakthrough[0] - stop_loss)
            elif index == 5:
                successful_trade_5.append(value_breakthrough[0] - stop_loss)
            elif index == 6:
                successful_trade_6.append(value_breakthrough[0] - stop_loss)
            elif index == 7:
                successful_trade_7.append(value_breakthrough[0] - stop_loss)
            elif index == 8:
                successful_trade_8.append(value_breakthrough[0] - stop_loss)
            elif index == 9:
                successful_trade_9.append(value_breakthrough[0] - stop_loss)
            elif index == 10:
                successful_trade_10.append(value_breakthrough[0] - stop_loss)
            elif index == 11:
                successful_trade_11.append(value_breakthrough[0] - stop_loss)
            elif index == 12:
                successful_trade_12.append(value_breakthrough[0] - stop_loss)
            elif index == 13:
                successful_trade_13.append(value_breakthrough[0] - stop_loss)
            elif index == 14:
                successful_trade_14.append(value_breakthrough[0] - stop_loss)
            elif index == 15:
                successful_trade_15.append(value_breakthrough[0] - stop_loss)
            elif index == 16:
                successful_trade_16.append(value_breakthrough[0] - stop_loss)
            elif index == 17:
                successful_trade_17.append(value_breakthrough[0] - stop_loss)


def check_diff_to_neckline_value(curr_extreme, type_of_double_formation):
    """Check percentage diff of extreme to neckline"""
    global neckline_value

    if type_of_double_formation == "0":
        if curr_extreme*0.98999 >= neckline_value: # 0.99499
            return True
    else:
        if curr_extreme*1.01001 <= neckline_value: # 1.00501
            return True

    return False

def check_previous_trend(chart_data, index_first_extreme, neckline_value, type_of_double_formation, len_of_formation):
    """Check trend before first extreme, DT=from down, BT=from up"""

    # nicht 9 sondern an langer zw Extrema anpassen
    previous_data = chart_data[['Close']][((chart_data.index.values < index_first_extreme) & 
    (chart_data.index.values >= (index_first_extreme - ceil(len_of_formation*1.5))))]['Close'].values

    signs = np.sign(previous_data - neckline_value)
    diff = np.diff(signs[signs != 0])

    if type_of_double_formation == "0":
        if len(diff[diff==2]) > 0:
            #print("Previous Data:", previous_data)
            #print("Neckline Val:", neckline_value)
            #print("Signs:", signs)
            #print("Diff of Signs:", diff[diff==2])
            return True
    else:
        if len(diff[diff==-2]) > 0:
            #print("Previous Data:", previous_data)
            #print("Index 1st Extr.:", index_first_extreme)
            #print("Length of f.:", len_of_formation)
            #print("Previous Data:", chart_data[['Close']][((chart_data.index.values < index_first_extreme) & 
            #(chart_data.index.values >= (index_first_extreme - ceil(len_of_formation*1.5))))])
            #print("Neckline Val:", neckline_value)
            #print("Signs:", signs)
            #print("Diff of Signs:", diff[diff==-2])
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
                print("Short gehen")
                print("BEI INDEX", index, "Nackenlinie durchbrochen")
                return index
        else:
            if val > neckline_value:
                found_necklines.append([neckline_value, start_neckline, end_neckline])
                print("\nDETECTED DOUBLE BOTTOM")
                print("Long gehen, bei Index", index, "Nackenlinie durchbrochen")
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
        return neckline_value * 1.02

    return neckline_value * 0.98

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
    fig_plot.savefig(fr'20220818_Neckline_Downfall_Uprise/2/DT/plot_formations_{company}.png')

main()