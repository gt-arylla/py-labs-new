#Function holds code to run prism data pulled from export CSVs in the C++ score code

#Imports
import pandas as pd

import os, sys, math, json, prism_fun
from glob import glob
from shutil import copyfile

#serial_dict is set up as:
#key - serial index.  Set arbitrarily
#value - binay string, which indicates which ROIs are printed and which aren't.  The length of the string must be the same length as the available ROIs

serial_dict={}
serial_dict[1]="101010101010101010110101010101"
serial_dict[2]="010101010101010101001010101010"

#optionally, combine ROIs
roi_combo_switch=1
roi_combo_list=[0,2,4,6,8,10,12,14,16,18]

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
export_list=[]
for file in files:
    print file
    run_switch=1
    filename=file.rpartition("\\")[2]

    try:
        df=pd.read_csv(file,header=0,error_bad_lines=False,warn_bad_lines=False)
        #iterate through all the ROIs, make a cotsu file, then put it into the detailed analysis
        #The photos are differentiated by 'set' in the following way:
        #'0' - blank image
        #'1' - print image
        #'-1' - blank roi.  However, this is on a serialized image so other elements of the image have been printed onto
        roi_index=1;
       
        while (run_switch):
            #print "ROI_index= %i" % (roi_index)
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
                        #print "ROI_index= %i" % (roi_idx)
                        qq={}
                        #export the prism data
                        try:
                            otsu_vector=prism_fun.make_otsu_vector(row,roi_idx)
                        except:
                            print "#######################OUTER ERROR#########################"
                            continue
                        
                        export_dict={}
                        export_dict["data"]=otsu_vector
                        #write the prism data so it's accessible by the c++ file
                        with open("prismdata.json","wb") as file:
                            file.write(json.dumps(export_dict))
                        #make the model accesible by the c++ file
                        try:
                            os.remove("modeldata.json")
                        except OSError:
                            pass
                        copyfile(filename+"_model_"+"1"+".json","modeldata.json")

                        os.system("C:\Users\gttho\Repos\Library\SaveMe\score.exe") #run scoring on exported json
                        score = float([line.rstrip('\n') for line in open('score.txt')][0])
                        print score

                        #Save the Mark information
                        df_columns=list(df)
                        mark=""
                        mark_type=""
                        if int(row["mark"])==0: 
                            mark=0
                            mark_type=0
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
                                    mark=1
                                    mark_type=1
                                    print_count+=1
                                else: 
                                    mark=0
                                    mark_type=-1
                                    blank_count+=1
                            elif "serial" in df_columns:
                                binary_string=serial_dict[row["serial"]]
                                binary_list=list(binary_string)
                                mark_value=int(binary_list[roi_idx])
                                #print row["serial"],binary_string, mark_value,roi_idx
                                if (mark_value):
                                    mark=1
                                    mark_type=1
                                    print_count+=1
                                else: 
                                    mark=0
                                    mark_type=-1
                                    blank_count+=1
                            else: 
                                mark=1
                                mark_type=1
                                print_count+=1

                        temp_dict={}
                        temp_dict["score"]=score
                        temp_dict["roi"]=roi_idx
                        temp_dict["model"]=filename
                        temp_dict["path"]=row["path"]
                        temp_dict["mark"]=mark
                        temp_dict["mark_type"]=mark_type
                        export_list.append(temp_dict)
                if roi_combo_switch: run_switch=0


            except:raise
    except: raise

#save as excel file
final_df=pd.DataFrame(export_list)
writer = pd.ExcelWriter('model_perf.xlsx')
final_df.to_excel(writer,'data')
writer.save()