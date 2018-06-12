#Identifies the best colorspaces and averages them together
#####################__________LIBRARIES______________################3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as pltfig
import os
import winsound
import sys
import time

import funcv

#############____________CONSTANTS_______________##########33
resize_big=0.7
resize_initial=0.2
resize_full=0.5
b_ratio=0.1105
resize_thresh=130000

file_type='.jpg'

if len(sys.argv)>=3:
    if sys.argv[2]=="png":
        file_type='.png'

c_thresh=.0
stdev_t=110
avg_thresh=60

eq_hist='n'
clay='n'
s3='n' 
r_check=0

#if user gave a path in the command line, use it.  Otherwise, use default
if len(sys.argv)>1:
	directory=sys.argv[1]
	directory=directory.replace("/","\\")
else:
	directory='\FigureImport'

#############3_______PHOTO IMPORT_______##############
#directory='G:\Developer\py-labs\FigureImport'
pat, dirs, files = os.walk(directory).next()
numfil=len(files)
total_counter=22*numfil #there are 22 colorspaces
dir_list=os.listdir(directory)

print directory
print len(sys.argv)

#show all colorspaces in the style of colorspace2.py
if len(sys.argv)<=3:
    for filename in dir_list:
        if filename[-4:].lower()==file_type:
            filname=os.path.join(directory, filename)
            img=cv2.imread(filname)

            if len(sys.argv)>2:
                if sys.argv[2]=="big" or sys.argv[2]=="png":

                    resize_big=1
                else:
                    resize_big=resize_initial
            img=cv2.resize(img,None,fx=resize_big,fy=resize_big,interpolation = cv2.INTER_CUBIC)

            pltfig.Figure(figsize=[120,100], dpi=200, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
            plt.subplot(461)
            i_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            plt.imshow(i_rgb)
            plt.title(filename,fontsize=8)
            plt.axis('off')
            for colorspace_index in range(22):
                i_ext,name=funcv.make_colorspace_single(img,colorspace_index)
                iter =colorspace_index+3
                plt.subplot(4,6,iter)
                plt.imshow(i_ext,cmap='jet')
                plt.axis('off')
                plt.title(str(colorspace_index)+" "+ name,fontsize=8)
            plt.show()
            break


tight_range=[120,175]
#default values are yes, no.  If user gives a third console input, use default values
if len(sys.argv)>2:
    thresh='n'
    eq_hist='n'
    clay='n'
    s3='n'
    pic_show=1
    final_cp_index=int(raw_input('Colorspace index? (int) '))
    start_value=int(raw_input('Start value? (int) '))
    end_value=int(raw_input('End value? (int) '))
    tight_range=[start_value,end_value]
else:
    pic_show=int(raw_input('Show Pic? 1/0 '))
    final_cp_index=int(raw_input('Colorspace index? (int) '))
    start_value=int(raw_input('Start value? (int) '))
    end_value=int(raw_input('End value? (int) '))
    tight_range=[start_value,end_value]
    #file=raw_input('Filename with extension: ')



for filename in dir_list:
    if filename[-4:].lower()==file_type:
        filname=os.path.join(directory, filename)
        # print filname
        #############_____PREP PHOTO__________###############3
        img=cv2.imread(filname)
        # print img
        # print img.shape
        # print img.size
        if img.size>resize_thresh and r_check==1:
        #resize=np.sqrt(resize_thresh/float(img.size))
        # print resize
            img=cv2.resize(img,None,fx=resize_full,fy=resize_full,interpolation = cv2.INTER_CUBIC)
        #b_ratio=b_ratio*resize
        # print b_ratio
			
        # pltfig.Figure(figsize=[12,10], dpi=200, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
        # plt.subplot(461)
        # i_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # plt.imshow(i_rgb)
        # plt.title(filename,fontsize=8)
        # plt.axis('off')

        ############________COLORSPACES_______#############
        i_ext,name=funcv.make_colorspace_single(img,final_cp_index)
        i_ext_temp=cv2.equalizeHist(i_ext)
        cv2.imwrite("export.jpg",i_ext_temp)
        cv2.imwrite("basis.jpg",img);
        plt.figure(5)
        plt.imshow(i_ext,clim=(tight_range[0],tight_range[1]),cmap="jet")
        plt.axis('off')
        plt.title(filename)
        if pic_show==1:
            plt.show()
        else:
            plt.savefig(filname+".png")
		