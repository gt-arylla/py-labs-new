#Function holds code to automatically run analysis of prism data contained in CSVs
#A lot of this work will be based on analysis loop in 'main_script.py' but it will be kept cleaner
#Also, this will ONLY be used to analyze prism data
#The data will be cast into the prism analysis in the form of 'costu_1.txt'

#We are replacing the 'make_otsu_index.py' call

#Imports
import pandas as pd

import os, sys, math, json, prism_fun
from glob import glob

#Make a serial map - maps basic ROI index to serialized ROI index
#For serial_dict[x]=[y,z]
#x = basic ROI Index
#y = Serial number
#z = Serialized ROI index
serial_dict={}
serial_dict[0]=[1,0]
serial_dict[1]=[2,0]
serial_dict[2]=[1,1]
serial_dict[3]=[2,1]
serial_dict[4]=[2,2]
serial_dict[5]=[1,2]
serial_dict[6]=[2,3]
serial_dict[7]=[1,3]
serial_dict[8]=[1,4]
serial_dict[9]=[2,4]
serial_dict[10]=[2,5]
serial_dict[11]=[1,5]

#Define bin constants
bins=["wh", "lg", "mg", "dg", "bk", "r", "o", "y", "sg", "g", "tu", "cy", "az", "b", "pu","ma","pi","bc"]
sides=["l","r"]

#Make list of CSV files
files=[]
start_dir='csv_import'
pattern   = "*.csv"
for dir,_,_ in os.walk(start_dir): 
    files.extend(glob(os.path.join(dir,pattern))) 

#prism_fun.serial_test(serial_dict)

#thresh,J,sen,spec=prism_fun.calculate_redundancy([0,2,5])
#print [thresh,J,sen,spec]


#Iterate through all the csv files
for file in files:
    try:
        df=pd.read_csv(file,header=0,error_bad_lines=False,warn_bad_lines=False)
        #iterate through all the ROIs, make a cotsu file, then put it into the detailed analysis
        #The photos are differentiated by 'set' in the following way:
        #'0' - blank image
        #'1' - print image
        #'-1' - blank roi.  However, this is on a serialized image so other elements of the image have been printed onto
        roi_index=7;
       
        while (True):
            print "ROI_index= %i" % (roi_index)
            try:         
                lf = open("otsu_0.txt", "wb")
                for index, row in df.iterrows():
                    qq={}
                    #export the prism data
                    printer=True
                    otsu_string=""
                    for bin in bins:
                        otsu_string+=" "+bin.upper()
                        for side in sides:
                            value=row["roi"+str(int(roi_index))+"_"+side+"_"+bin]
                            #don't save line if there are nan values
                            if math.isnan(value): printer=False
                            otsu_string+=" "+str(value)
                        otsu_string+="\n"
                    if not printer: continue
                    qq["otsu"]=otsu_string

                    #export the filename
                    fn=row["path"]
                    qq["file"]=fn

                    #export the set
                    df_columns=list(df)
                    pset=""
                    if int(row["mark"])==0: pset="0"
                    elif int(row["mark"])==1:
                        if "serial" in df_columns:
                            active_serial=serial_dict[roi_index][0]
                            if row["serial"]==active_serial: pset="1"
                            else: pset="-1"
                        else: pset="1"
                    else: continue
                    qq["set"]=pset

                    #write to text file
                    lf.write(json.dumps(qq) + "\n")

                lf.close()

                #perform logistic regression analysis
                os.system("python make_feature_maps.py") #make giant 630x61x61 index
                print "DONE"
                os.system("python fit_coeffs.py") #fit the data to a logistic regression model
                os.system("python eval_test_perf.py") #determine the accuracy of the model

                #rename important files so they won't be overwritten later
                os.rename('coeffs.txt', 'coeffs_'+str(int(roi_index))+'.txt')
                os.rename('transcript.txt', 'transcript_'+str(int(roi_index))+'.txt')
                os.rename('testperf.txt', 'testperf_'+str(int(roi_index))+'.txt')
                os.rename('master_map.json', 'master_map_'+str(int(roi_index))+'.json')
                roi_index+=1
                
            except:
                raise
    except:
        raise