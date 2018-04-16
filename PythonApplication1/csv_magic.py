#import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as pltfig
#import qrtools
import os
import winsound
import sys
import time
import csv
import math

def combine_csv(directory,output_csv_name):

    ####___Code_____######
    #Set up file export
    saveDirectory=directory
    #Set up auto csv import
    #cwd = os.getcwd()
    #directory=cwd+directory
    pat, dirs, files = os.walk(directory).next()
    numfil=len(files)
    export_data_list=[]
    
    for filename in os.listdir(directory):

        csv_name=filename[:-4]
        print csv_name
        csv_file=directory+csv_name+".csv"
        print csv_file
        #Put CSV data into Mat
        with open(csv_file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                export_data_list.append(row)
            #    print row
            #    try:
            #        line_export=''
            #        for val in row:
            #            line_export=line_export+val+","
            #            #print line_export
            #    except:
            #        print 'line import failed'
            #export_data_list.append(line_export)
        print "Finished CSV Import."

    #for line in np.arange(len(export_data_list)):
    #    for val in line:
    #        print val+",",
    #    print 
    #with open(output_csv_name+'.csv', 'a+') as newFile:
    #    newFileWriter = csv.writer(newFile)
    #    for line in np.arange(len(export_data_list)):
    #        newFileWriter.writerow([export_data_list[line]])

    with open(output_csv_name+'.csv', 'wb') as newFile:
        newFileWriter = csv.writer(export_data_list)