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

	# Polyregression der Close-Daten, curve-fitting
	#polyfit = np.polyfit(datasetIndex, datasetCloseVal, 5)
	#yvalpoly = np.polyval(polyfit, datasetIndex)
	#print(yvalpoly)
	fig1 = plt.figure(figsize=(80, 40), dpi= 120, facecolor='w', edgecolor='k')
	
	# get indizes of maxs and mins
	arrIndexMaxima = sp.argrelextrema(np.array(datasetCloseVal), np.greater)
	arrIndexMinima = sp.argrelextrema(np.array(datasetCloseVal), np.less)
	#print("Maximum ")
	#print(arrIndexMaxima[0])
	#print(arrIndexMinima[0])

	arrValuesMaxima = datasetCloseVal.iloc[arrIndexMaxima[0]]['Close'].values
	arrValuesMinima = datasetCloseVal.iloc[arrIndexMinima[0]]['Close'].values

	#print(arrValuesMaxima)
	#print(arrValuesMinima)

	# get values of minimums/maximums
	max_vals = []
	max_vals_index = []
	max_values = []
	min_vals = []
	min_vals_index = []
	plt.xticks(range(0, df.size))
	#print(np.where(df.index.values > 5, df.index.values))
	
	
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
				#print(rangeArr) # get bis i+5 points
				#print(valuesOfRange)
				necklineValue = np.min(valuesOfRange[:-4])[0] # min
				startNeckline = min(rangeArr)*(1/datasetCloseVal.size)
				endNeckline = (max(rangeArr))*(1/datasetCloseVal.size)
				plt.axhline(y=necklineValue, xmin=startNeckline, xmax=endNeckline, color='red', alpha=0.1)
				
				# wenn unter Neckline faellt, dann ists DoubleTop
				
				firstIndexBelowNeckline = getFirstIndexBreakingNeckline(valuesOfRange[-4:]['Close'].values, necklineValue, "<")
				if firstIndexBelowNeckline > -1:	
					indexBreakthrough = arrIndexMaxima[0][index] + firstIndexBelowNeckline
					valueBreakthrough = datasetCloseVal[df.index.values == indexBreakthrough]['Close'].values
					#print(valueBreakthrough)
					plt.plot(indexBreakthrough, valueBreakthrough, 'o', alpha=1, color='blue')
					plt.plot(valuesOfRange.index.values, valuesOfRange.values, '-', color='green', markersize=4, alpha=1)
					print("Detected Double Top:")
					print(valuesOfRange.index.values)
	print("")
	print("DOUBLE BOTTOM")
	# doubleBottom
	for index, i in enumerate(arrIndexMinima[0]):
		if index > 0:
			if (arrValuesMinima[index-1]>=arrValuesMinima[index]) and math.isclose(arrValuesMinima[index-1],arrValuesMinima[index],rel_tol=0.02):
				print("I-1 ", arrIndexMinima[0][index-1], "I ", arrIndexMinima[0][index])
				min_vals.append(arrValuesMinima[index-1])
				min_vals_index.append(arrIndexMinima[0][index-1])
				min_vals.append(arrValuesMinima[index])
				min_vals_index.append(i)
				rangeBetweenTwoMins = (df.index.values >= arrIndexMinima[0][index-1]) & (df.index.values < arrIndexMinima[0][index]+5)
				arrRangeMins = df.index.values[rangeBetweenTwoMins]
				arrValuesOfMins = datasetCloseVal[rangeBetweenTwoMins]
				print(arrValuesOfMins)
				necklineValue = np.max(arrValuesOfMins[:-4])[0] #max
				startNeckline = min(arrRangeMins)*(1/datasetCloseVal.size)
				endNeckline = max(rangeArr)*(1/datasetCloseVal.size)

				firstIndexBreakingNeckline = getFirstIndexBreakingNeckline(arrValuesOfMins[-4:]['Close'].values, necklineValue, ">")
				if firstIndexBreakingNeckline > -1:
					indexBreakthrough = arrIndexMinima[0][index] + firstIndexBreakingNeckline
					valueBreakthrough = datasetCloseVal[df.index.values == indexBreakthrough]['Close'].values
					print("Detected Double Bottom: ")
					print(arrValuesOfMins.index.values)

	# Plot data Points
	plt.plot(datasetIndex, datasetCloseVal, '-', markersize=1.5, color='black', alpha=0.6)
	#plt.plot(datasetIndex, yvalpoly, label='fit')
	plt.plot(max_vals_index, max_vals, 'o', markersize=9.5, color='green')
	#plt.plot(arrIndexMinima[0], min_vals, 'o', markersize=9.5, color='red')
	#plt.xticks(np.arange(len(datasetCloseVal)), np.arange(1, len(datasetCloseVal)+1))
	plt.xlim([0, datasetCloseVal.size])
	fig1.savefig("test2.png")

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