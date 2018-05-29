#####################__________LIBRARIES______________################3
import os
import shutil
import csv

#############3_______PHOTO IMPORT_______##############
directory_move="FILE_master_badPics"

csv_file="Pic_import.csv"

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