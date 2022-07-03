# Identification of formations

This repository contains a Python Script for detecting and trading of Double Top & Double Bottom formations. I focused on analyzing companies listed on the Nasdaq. The chartdata of the companies were given as csv files.
I used Python because of its strong and efficient data analysis capability.I used the library Pandas for reading the given csv files of the companies. Numpy and Scipy were used for analyzing and detecting the formations in the data. The charts and the detected formations are plotted with functions of the library Matplotlib.  


As the project is not finished yet, some functions are commented out for development reasons.
I plan to implement other features in the future. For example I want to integrate the detection of Triple Top/Bottom formations.
The file 'Masterprojekt_Poster_Ghukasyan.pdf' contains further information about this project (i.e. motivation and theoretical explanation of Double Top formations)

## How to install and run the project

To run the project, you need to install python and the following libraries:
    - pandas
    - numpy
    - scipy
    - matplotlib
    
Additionally you need also access to financial data. As the code is developed for csv source files, you have to customize the code for the source files you use.