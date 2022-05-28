import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
import scipy.signal as sp
import math

def main():

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

	args = my_parser.parse_args()

	company = args.c.upper()
	time_frame = args.t.lower()	
	df = readFile(company, time_frame)
	detectDoubleTop(df)	

def readFile(company, time_frame):

	filepath = r'/data/financedata/2020ss/gelling/data/kibot3/NASDAQ/{}/{}.candle.{}.unadjusted'.format(company, company.lower(), time_frame)
	df = pd.read_csv(
		filepath,
		",",
		header=None, 
		names=["Date","Time","Open", "High","Low","Close","Umsatz (St)", "Quelle", "Import-Datum"],
		parse_dates=[0])
	
	return df

def detectDoubleTop(df):

	df = df[:300]
	datasetIndex = df.index.tolist()
	datasetCloseVal = df[['Close']]

	fig1 = plt.figure(figsize=(80, 40), dpi= 120, facecolor='w', edgecolor='k')
	
	# get indizes of maxs and mins
	arrIndexMaxima = sp.argrelextrema(np.array(datasetCloseVal), np.greater)
	arrIndexMinima = sp.argrelextrema(np.array(datasetCloseVal), np.less)

	arrValuesMaxima = datasetCloseVal.iloc[arrIndexMaxima[0]]['Close'].values
	arrValuesMinima = datasetCloseVal.iloc[arrIndexMinima[0]]['Close'].values

	# get values of minimums/maximums
	max_vals = []
	max_vals_index = []
	max_values = []
	min_vals = []
	min_vals_index = []
	plt.xticks(range(0, df.size))	
	
	# index = Index innerhalb der Index-Schleife;i = Index wo Maximum vorliegt
	for index, i in enumerate(arrIndexMaxima[0]):		
		if index > 0:	
			# aufeinanderfolgende Tops ca gleich hoch (1 Prozent Abweichung) -> Anfindex eintragen
			if (arrValuesMaxima[index-1]>=arrValuesMaxima[index]) and math.isclose(arrValuesMaxima[index-1],arrValuesMaxima[index],rel_tol=0.02):
				
				plt.axvline(x=i, color='black', markersize=0.1, alpha=0.3)
				plt.axvline(x=arrIndexMaxima[0][index-1], color='black', markersize=0.1, alpha=0.3) # i stimmt nicht mit -1
				#print("I-1 ",arrIndexMaxima[0][index-1],"I ", arrIndexMaxima[0][index])
				max_vals.append(arrValuesMaxima[index-1])
				max_vals_index.append(arrIndexMaxima[0][index-1])
				max_vals.append(arrValuesMaxima[index])
				max_vals_index.append(i)
				
				rangeBetweenTwoMaxs = (df.index.values >= arrIndexMaxima[0][index-1]) & (df.index.values < arrIndexMaxima[0][index]+5)
								
				rangeArr = df.index.values[rangeBetweenTwoMaxs]
				valuesOfRange = datasetCloseVal[rangeBetweenTwoMaxs]
				necklineValue = np.min(valuesOfRange[:-4])[0] # min
				startNeckline = min(rangeArr)*(1/datasetCloseVal.size)
				endNeckline = max(rangeArr)*(1/datasetCloseVal.size)
				plt.axhline(y=necklineValue, xmin=startNeckline, xmax=endNeckline, color='red', alpha=0.1)
				
				# wenn unter Neckline faellt, dann ists DoubleTop
				
				firstIndexBelowNeckline = getFirstIndexBreakingNeckline(valuesOfRange[-4:]['Close'].values, necklineValue, "<")
				if firstIndexBelowNeckline > -1:	
					indexBreakthrough = arrIndexMaxima[0][index] + firstIndexBelowNeckline
					valueBreakthrough = datasetCloseVal[df.index.values == indexBreakthrough]['Close'].values
					plt.plot(indexBreakthrough, valueBreakthrough, 'o', alpha=1, color='blue')
					plt.plot(valuesOfRange.index.values, valuesOfRange.values, '-', color='green', markersize=4, alpha=1)
					print("Detected Double Top:")
					print(valuesOfRange.index.values)
	print("")
	print("DOUBLE BOTTOM")
	# doubleBottom iterate through all minima
	for index, i in enumerate(arrIndexMinima[0]):
		if index > 0:
			# if prev minima bigger eq curr minima & rel_diff 2 percent
			if (arrValuesMinima[index-1]<=arrValuesMinima[index]) and math.isclose(arrValuesMinima[index-1],arrValuesMinima[index],rel_tol=0.02):
				
				min_vals.append(arrValuesMinima[index-1]) # prev minimum
				min_vals_index.append(arrIndexMinima[0][index-1]) # index of pre minimum
				min_vals.append(arrValuesMinima[index]) # curr minimum
				min_vals_index.append(i) # index of curr minimum
				
				# rangetoexamine
				rangeBetweenTwoMins = (df.index.values >= arrIndexMinima[0][index-1]) & (df.index.values < arrIndexMinima[0][index]+5)
				arrRangeMins = df.index.values[rangeBetweenTwoMins]
				arrValuesOfMins = datasetCloseVal[rangeBetweenTwoMins]
				necklineValue = np.max(arrValuesOfMins[:-4])[0] 
				startNeckline = min(arrRangeMins)*(1/datasetCloseVal.size)
				endNeckline = max(rangeArr)*(1/datasetCloseVal.size)

				firstIndexBreakingNeckline = getFirstIndexBreakingNeckline(arrValuesOfMins[-4:]['Close'].values, necklineValue, ">")
				if firstIndexBreakingNeckline > -1:
					#print(arrValuesOfMins)
					indexBreakthrough = arrIndexMinima[0][index] + firstIndexBreakingNeckline
					valueBreakthrough = datasetCloseVal[df.index.values == indexBreakthrough]['Close'].values
					#print("Detected Double Bottom: ")
					#print(arrValuesOfMins.index.values)

	# Plot data Points
	plt.plot(datasetIndex, datasetCloseVal, '-', markersize=1.5, color='black', alpha=0.6)
	plt.plot(max_vals_index, max_vals, 'o', markersize=9.5, color='green')
	plt.xlim([0, datasetCloseVal.size])
	fig1.savefig("test2.png")

	detectDoubleFormation(arrIndexMaxima[0], arrValuesMaxima, 0, df)

def detectDoubleFormation(arrIndexExtremeValues, arrValsExtremeValues, typeOfDoubleFormation, df):
	# 0 = Double Top; 1 = Double Bottom; 2 = Both
	
	datasetCloseVal = df[['Close']]

	critCompareExtremeVals = '>=' #(prevExtreme >= currExtreme)
	necklineOperator = "<"
	if typeOfDoubleFormation == 1:
		critCompareExtremeVals = '<=' #(prevExtreme <= currExtreme)
		necklineOperator = ">"

	for indexArr, indexDataset in enumerate(arrIndexExtremeValues):
		if indexArr > 0:
			prevExtreme = arrValsExtremeValues[indexArr-1]
			currExtreme = arrValsExtremeValues[indexArr]
			
			if (eval(str(prevExtreme) + critCompareExtremeVals + str(currExtreme)) and math.isclose(prevExtreme, currExtreme, rel_tol=0.02)):
				
				conditionRangeBetweenTwoExtremes = (df.index.values >= arrIndexExtremeValues[indexArr-1]) & (df.index.values < arrIndexExtremeValues[indexArr]+5)
				
				arrRangeExtremes = df.index.values[conditionRangeBetweenTwoExtremes]
				arrValuesOfExtremes = datasetCloseVal[conditionRangeBetweenTwoExtremes]
				necklineValue = np.max(arrValuesOfExtremes[:-4])[0]
				firstIndexBreakingNeckline = getFirstIndexBreakingNeckline(arrValuesOfExtremes[-4:]['Close'].values, necklineValue, necklineOperator)
				if firstIndexBreakingNeckline > -1: 
					indexBreakthrough = currExtreme + firstIndexBreakingNeckline
					valueBreakthrough = datasetCloseVal[df.index.values == indexBreakthrough]['Close']
					print("Detected Double Formations: ")
					print(arrValuesOfExtremes.index.values)


def getFirstIndexBreakingNeckline(valueArr, necklineValue, operator):
	if operator == "<":
		outerCondition = all(val < necklineValue for val in valueArr)
	elif operator == ">":
		outerCondition = all(val > necklineValue for val in valueArr)
		
	if not outerCondition:
		for index, val in enumerate(valueArr):
			if operator == "<": 	
				if val < necklineValue:
					return index+1
			elif operator == ">":
				if val > necklineValue:
					return index+1
	return -1

main()