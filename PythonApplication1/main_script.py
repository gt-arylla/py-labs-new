#Run Functions
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fun
import copy
import plotter
print 'running...'

if (1): ##FAF plotting
    #Define Paths
    directory="G://Developer//csvExports//"
    csv_name="171020 Area and Erosion Sweep.csv"
    csv_file=directory+csv_name
    plotter.FAF_plotter(csv_file,[1],[7],[3,0,2])

