#Allows for user-friendly sorting
#####################__________LIBRARIES______________################3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import winsound
import sys
import copy
import shutil

#############3_______PHOTO IMPORT_______##############

directory_move="C://Users//gttho//Resilio Sync//Developer//PhotoExports//180405 KG TB Grids//"

directory_start="C://Users//gttho//Resilio Sync//Developer//PhotoExports//180405 KG TB Grids//sortme"
#directory_start="G://Developer//PhotoExports//180312 JB_FF_v1//"
sort_dict={"1":"True//Perfect","2":"True//Bloated","3":"Fail//External Contour","4":"Fail//Hole in Contour","5":"Fail//NoCentaur","6":"Fail//Part Contour"}
sort_dict={"1":"good","0":"bad"}
adder_string=["50mS","75mS","100mS"]
adder_string=[""]
for adder in adder_string:
    all_file_switch=0

    if not all_file_switch:
        directory=directory_start+adder
        pat, dirs, files = os.walk(directory).next()
        numfil=len(files)
        total_counter=22*numfil #there are 22 colorspaces
        files_to_scan=os.listdir(directory)

    ##Scan ALL Files in Tree, rather than specific tree
    
    if(all_file_switch):
        files = []
        start_dir=directory_start

        pattern = "*.jpg"
        
        for dir,_,_ in os.walk(start_dir):
            
            files.extend(glob(os.path.join(dir,pattern))) 
        files_to_scan=files
    

    ##############__________PREP HOLDING ARRAYS_______############3
    date='dummy'
    counter1=0 #use for the total counter, to input values into the coeffs array
    finder_counter=0

    

    for filename in files:
        #if (filename[-4:].lower()=='.jpg') and (filename[:9]=='PE_Legacy'):
        #print filename
    
        if (filename[-4:].lower()=='.png'):
            #try:
            if True:
               

                #Load Image
                if all_file_switch:
                    filname=filename
                else:
                    filname=os.path.join(directory, filename)
                print filname

                # #Show image
                img=cv2.imread(filname)
                ##print img
                #plt.ion()
                #plt.imshow(img)
                #plt.show()
                #plt.draw()
                #plt.pause(0.001)

                cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN);
                cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('image',img)
                cv2.waitKey(100)

                folder=raw_input("Photo Folder Index")

                final_move_directory=directory_move+"//"+sort_dict[folder]+"//"

                shutil.move(os.path.join(directory,filename),os.path.join(final_move_directory,filename))

                
               
               