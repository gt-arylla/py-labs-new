#Function holds code to automatically run performance analysis of .txt files
#the time consuming step is everything before this

#Imports
import pandas as pd
import os, sys, math, json, prism_fun
from glob import glob

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
    run_switch=1
    filename=file.rpartition("\\")[2]
    data={}
    try:
        os.system("python eval_test_perf.py "+'"'+filename+'"'+"_ _0") #determine the accuracy of the model
        with open ("testperf.txt") as json_file:
            data=json.load(json_file)
            data["filename"]=filename
            export_list.append(data)
    except:
        continue

#save as excel file
final_df=pd.DataFrame(export_list)
writer = pd.ExcelWriter('perf.xlsx')
final_df.to_excel(writer,'data')
writer.save()