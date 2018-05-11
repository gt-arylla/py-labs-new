#Identifies the best colorspaces and averages them together
#####################__________LIBRARIES______________################3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as pltfig
import os
import winsound
import sys
from glob import glob
import time
#import piexif
#import piexif.helper
import string
import copy
import shutil
import csv

#############____________CONSTANTS_______________##########33

user_comment='test_comment'
#file=raw_input('Filename with extension: ')

#############3_______PHOTO IMPORT_______##############

directory_move="C://Users//gttho//Repos//py-labs-new//PythonApplication1//FILE_master_badPics"

directory_start="C://Users//gttho//Documents//Visual Studio 2017//Projects//PythonApplication1//PythonApplication1//bad_pics"
directory_start="C://Users//gttho//Resilio Sync//Original Images//CanadaGoose//180325-180326 DC_Wash_Conc_Sweep"
directory_start="C://Users//gttho//Resilio Sync//Original Images//CanadaGoose//180330 Refined Conc Sweep Test"
adder_string=["50mS","75mS","100mS"]
adder_string=[""]

csv_file="C://Users//gttho//Repos//py-labs-new//PythonApplication1//Pic_import.csv"

with open(csv_file,'rb') as f:
    reader=csv.reader(f)
    lst=list(reader)


for row in lst:
    for col in row:
        col_split=col.split("\\")
        print col
        print col_split[-1]
        file_move=os.path.join(directory_move,col_split[-1])
        shutil.copyfile(col,file_move)
print lst[0][0]