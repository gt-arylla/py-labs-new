#Function holds code to automatically run analysis of prism data contained in CSVs
#A lot of this work will be based on analysis loop in 'main_script.py' but it will be kept cleaner
#Also, this will ONLY be used to analyze prism data
#The data will be cast into the prism analysis in the form of 'costu_1.txt'

#We are replacing the 'make_otsu_index.py' call

#Imports
import pandas as pd

import os, sys, math, json, prism_fun,fun
from glob import glob

#serial_dict is set up as:
#key - serial index.  Set arbitrarily
#value - binay string, which indicates which ROIs are printed and which aren't.  The length of the string must be the same length as the available ROIs

serial_dict={}
serial_dict[1]="101010101010101010110101010101"
serial_dict[2]="010101010101010101001010101010"

#optionally, combine ROIs
roi_combo_switch=1
roi_combo_list=[0,2,4,6,8,10,12,14,16,18]
#roi_combo_list=[6,8,10]
#roi_combo_list=[12,14,16]

#Make list of CSV files
files=[]
start_dir='csv_import'
pattern   = "*.csv"
for dir,_,_ in os.walk(start_dir): 
    files.extend(glob(os.path.join(dir,pattern))) 

#prism_fun.model_header_compile(range(0,12))

#prism_fun.serial_test(serial_dict,False)
#thresh,J,sen,spec=prism_fun.calculate_redundancy([0,2,5])
#print [thresh,J,sen,spec]


#Iterate through all the csv files
print files
for file in files:
    run_switch=1
    filename=file.rpartition("\\")[2]

    try:
        df=pd.read_csv(file,header=0,error_bad_lines=False,warn_bad_lines=False)
        #df=fun.arbitrary_include(df,"color","red")
        #iterate through all the ROIs, make a cotsu file, then put it into the detailed analysis
        #The photos are differentiated by 'set' in the following way:
        #'0' - blank image
        #'1' - print image
        #'-1' - blank roi.  However, this is on a serialized image so other elements of the image have been printed onto
        roi_index=0;
       
        while (run_switch):
            print "ROI_index= %i" % (roi_index)
            try:         
                lf = open("otsu_0.txt", "wb")
                blank_count=0
                print_count=0
                for index, row in df.iterrows():
                    if roi_combo_switch:
                        roi_list=roi_combo_list
                    else:
                        roi_list=[roi_index]
                    for roi_idx in roi_list:
                        qq={}
                        #export the prism data
                        try:
                            otsu_string=prism_fun.make_otsu_string(row,roi_idx)
                        except:
                            #print "#######################OUTER ERROR#########################"
                            continue
                        qq["otsu"]=otsu_string

                        #export the filename
                        fn=row["path"]
                        qq["file"]=fn

                        #export the set
                        df_columns=list(df)
                        pset=""
                        if int(row["mark"])==0: 
                            pset="0"
                            blank_count+=1
                        elif int(row["mark"])==1:
                            if ("decimal" in df_columns) and ("roi_count" in df_columns):
                                #Get identity of ROI via the 'decimal' input
                                #decimal is converted to binary, which is then converted to a vector of bools
                                roi_number=row["roi_count"]
                                decimal_print=row["decimal"]
                                binary_string=format(decimal_print,'b').zfill(roi_number)
                                binary_list=list(binary_string)
                                mark_value=int(binary_list[roi_idx])
                                if (mark_value): 
                                    pset="1"
                                    print_count+=1
                                else: 
                                    pset="-1"
                                    blank_count+=1
                            elif "serial" in df_columns:
                                binary_string=serial_dict[row["serial"]]
                                binary_list=list(binary_string)
                                mark_value=int(binary_list[roi_idx])
                                #print row["serial"],binary_string, mark_value,roi_idx
                                if (mark_value):
                                    pset="1"
                                    print_count+=1
                                else: 
                                    pset="-1"
                                    blank_count+=1
                            else: 
                                pset="1"
                                print_count+=1
                        else: continue
                        qq["set"]=pset

                        #write to text file
                        lf.write(json.dumps(qq) + "\n")

                lf.close()

                if blank_count==0 or print_count==0:
                    print "no data"
                    roi_index+=1
                    continue
                print blank_count
                print print_count
                #perform logistic regression analysis
                os.system("python make_feature_maps.py") #make giant 630x61x61 index
                os.system("python fit_coeffs.py") #fit the data to a logistic regression model
                os.system("python eval_test_perf.py") #determine the accuracy of the model

                #rename important files so they won't be overwritten later
                os.rename('coeffs.txt', filename+'_coeffs_'+str(int(roi_index))+'.txt')
                os.rename('transcript.txt', filename+'_transcript_'+str(int(roi_index))+'.txt')
                os.rename('testperf.txt', filename+'_testperf_'+str(int(roi_index))+'.txt')
                os.rename('master_map.json', filename+'_master_map_'+str(int(roi_index))+'.json')
                if roi_combo_switch: run_switch=0
                roi_index+=1
                
            except KeyError:
                break
            except WindowsError:
                raise
        #prism_fun.serial_test(serial_dict)
    except:
        continue