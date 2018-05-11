#Run Functions
print "importing..."
#import cv2
import numpy as np
print 'plt import'
import matplotlib.pyplot as plt
print 'other import1'
import fun
import copy
import plotter
import csv_magic
print 'other import2'
import sys
import os
import csv
print 'other import3'
from glob import glob
import time
import copy
print 'other import4'
import pandas as pd
import itertools as it
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
print 'other import5'
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import time
import string
import json

print 'running...'

if (0): ##FAF plotting
    #Define Paths
    directory="G://Developer//csvExports//"
    tn=4
    if tn==0:
        csv_name="171103_SatCctCb_ps7_combo.csv"
        csv_file=directory+csv_name
        #plotter.FAF_plotter(csv_file,[6],[9],[7,0,1,2,3])
        #for j in np.arange(55,109):
        #    for i in np.arange(174,175):
        #plotter.FAF_plotter(csv_file,[415],[285],[169,165],yaxis=[100,125],xaxis=[4000,9000])
        plotter.FAF_plotter(csv_file,[229],[441],[499,495],yaxis=[4000,9000],xaxis=[0,100])
    if tn==1:
        csv_name="171103 ThicknessNew.csv"
        #csv_name="171103_SuperData_ps6and7_HueOt_center.csv"
        csv_file=directory+csv_name
        plotter.FAF_plotter(csv_file,[1],[0],[2,5],yaxis=[0,7.2e7],xaxis=[-1,17])
    if tn==2:
        csv_name="cbrgb_2_noHeader.csv"
        #csv_name="171103_SuperData_ps6and7_HueOt_center.csv"
        csv_file=directory+csv_name
       # plotter.FAF_plotter(csv_file,[382],[297],[334,332],yaxis=[100,125],xaxis=[4000,9000])
       # plotter.FAF_plotter(csv_file,[337],[382],[333,332],yaxis=[4000,9000],xaxis=[-10,110])
       # plotter.FAF_plotter(csv_file,[337],[297],[334,332],yaxis=[100,125],xaxis=[-10,110])
        plotter.FAF_plotter(csv_file,[333],[297],[334,332],yaxis=[100,125],xaxis=[0,3600])
    if tn==3:
        directory="C://Users//gttho//Documents//Visual Studio 2017//Projects//PythonApplication1//PythonApplication1//"
        csv_name="cbrgb_4.csv"
        #csv_name="171103_SuperData_ps6and7_HueOt_center.csv"
        csv_file=directory+csv_name
       # plotter.FAF_plotter(csv_file,[382],[297],[334,332],yaxis=[100,125],xaxis=[4000,9000])
       # plotter.FAF_plotter(csv_file,[337],[382],[333,332],yaxis=[4000,9000],xaxis=[-10,110])
       # plotter.FAF_plotter(csv_file,[337],[297],[334,332],yaxis=[100,125],xaxis=[-10,110])
        #for y_col in [0,1,39,96,205,261,341,387,388,390,392,394,395,453]:
        plotter.FAF_plotter(csv_file,[452],[341],[455,457])
    if tn==4:
        directory="G://Google Drive//Datasets//"
        csv_name="171123_TriPhotoTest_train_demo.csv"
        csv_name="171130_TriPhotoTest_NM1_NM2_P_M_R_2.csv"
        csv_name="171130_TriPhotoTest_NM1_NM2_AP_M_R.csv"
        csv_name="171207_WbExpTest_NM1_NM2_P_M_R.csv"
        #csv_name="171103_SuperData_ps6and7_HueOt_center.csv"
        csv_file=directory+csv_name
       # plotter.FAF_plotter(csv_file,[382],[297],[334,332],yaxis=[100,125],xaxis=[4000,9000])
       # plotter.FAF_plotter(csv_file,[337],[382],[333,332],yaxis=[4000,9000],xaxis=[-10,110])
       # plotter.FAF_plotter(csv_file,[337],[297],[334,332],yaxis=[100,125],xaxis=[-10,110])
        #for y_col in [0,1,39,96,205,261,341,387,388,390,392,394,395,453]:
        for x in [102,103]:
            for y in [[66,71]]:
                plotter.FAF_plotter(csv_file,[x],[66,71],[1,2,3])
    if tn==5:
        directory="G://Developer//csvExports//"
        csv_name="171129_SuperDuperSet_difference.csv"
        #csv_name="171103_SuperData_ps6and7_HueOt_center.csv"
        csv_file=directory+csv_name
       # plotter.FAF_plotter(csv_file,[382],[297],[334,332],yaxis=[100,125],xaxis=[4000,9000])
       # plotter.FAF_plotter(csv_file,[337],[382],[333,332],yaxis=[4000,9000],xaxis=[-10,110])
       # plotter.FAF_plotter(csv_file,[337],[297],[334,332],yaxis=[100,125],xaxis=[-10,110])
        #for y_col in [0,1,39,96,205,261,341,387,388,390,392,394,395,453]:
        for x in [45,46]:
            plotter.FAF_plotter(csv_file,[x],[1,44],[47])
if (0): #cotsu DataFrame Plotting
    csv_file="G://Google Drive//Datasets//180122_cOtsu_till_App36.csv"
    plotter.cotsu_plotter(csv_file)
if (0): #PivotTable Histogram
    csv_file="G://Developer//csvExports//180124 KM Exports//180130//KM_Noise_100x_180130_-5.csv"
    plotter.pivot_histogram(csv_file,'Tag','Match')
if(0): #csv multi file to single file
    #define function inputs
    directory="C://Users//gttho//Documents//Visual Studio 2017//Projects//PythonApplication1//PythonApplication1//Resources//csv_import//"
    output_csv_name="combo_result2"
    csv_magic.combine_csv(directory,output_csv_name)
if(0): #Auto-photo-import.  To Return EXIF data or do analysis
    #############3_______PHOTO IMPORT_______##############
    directory='G://Developer//py-labs//FigureImport'
    directory='G://Google Drive//Original Images//Embroidery//171111 New App Photo//'
    directory="C://Users//gttho//Offline//Arylla Temp Photos//171114 Single Strand Photos-20171114T181618Z-001//171114 Single Strand Photos//AllPics//cnt2"
    directory="G://Google Drive//Original Images//Embroidery//171122 TriPhoto Test//"
    pat, dirs, files = os.walk(directory).next()
    numfil=len(files)
    pn=1
    for filename in os.listdir(directory):
        filname=os.path.join(directory, filename)
        if pn==1:
            img=cv2.imread(filname)
            img_cct=fun.cct(img)
            plt.imshow(img_cct)
            plt.show()
            plt.cla
        if pn==2:
            print directory,
            print filname
            fun.print_exif_UC(filname)
if(0): #Auto-photo-import.  Test new mask algo
    #############3_______PHOTO IMPORT_______##############
    directory='C://Users//gttho//Source//Repos//Library//SaveMe'
    pat, dirs, files = os.walk(directory).next()
    numfil=len(files)
    mat_list=[]
    count=0
    for filename in os.listdir(directory):
        if filename[-4:]=='.jpg':
            filname=os.path.join(directory, filename)
            print filname
            img_start=cv2.imread(filname)
            #plt.imshow(img_start)
            #plt.show()
            img=copy.copy(img_start)
            #img_cb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
            mat_list=[]
            for color in range(0,3):
                if color==0:
                    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
                    img_grey=img_grey[:,:,2]
                elif color==1:
                    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                    img_grey=img_grey[:,:,0]
                elif color==2:
                    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
                    img_grey=img_grey[:,:,0]
                img_dims=img_grey.shape
                img_crop=img_grey[int(img_dims[0]*0.3333):int(img_dims[0]*0.666),int(img_dims[1]*0.3333):int(img_dims[1]*0.666)]
                img_otsu_val=cv2.threshold(img_crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
                img_otsu=cv2.threshold(img_grey,img_otsu_val,255,cv2.THRESH_BINARY)[1]
                if color==0 or color==1:
                    img_otsu=cv2.bitwise_not(img_otsu)
                if color==2:
                    img_otsu=cv2.threshold(img_grey,254,255,cv2.THRESH_BINARY)[1]
                mat_list.append(img_otsu)
                print color
                #plt.imshow(img_otsu)
                #plt.show()
            mat_and_list=[]
            mat_and_list.append(cv2.bitwise_and(mat_list[0],mat_list[1]))
            mat_and_list.append(cv2.bitwise_and(mat_list[2],mat_list[1]))
            mat_and_list.append(cv2.bitwise_and(mat_list[0],mat_list[2]))
            mat_fin=cv2.bitwise_or(mat_and_list[0],mat_and_list[1])
            mat_fin=cv2.bitwise_or(mat_fin,mat_and_list[2])
            #mat_fin2=fun.area_filter(copy.copy(mat_fin),10000)
            #plt.subplot(121)
            #plt.imshow(mat_fin)
            #plt.subplot(122)
            #plt.imshow(mat_fin)
            #plt.show()
            kernel = np.ones((13,13),np.uint8)
            closing = cv2.morphologyEx(mat_fin, cv2.MORPH_OPEN, kernel)
            #plt.imshow(closing)
            #plt.show()

            cv2.imwrite("export2_"+filename,closing)
            del mat_and_list
            del mat_fin
            del img
            del img_start
            #except:
            #    print 'error'
            count=count+1
            print count/float(numfil)
if (0): #Multi-photo modeling
    tn=10
    if tn==4:
        csv_file='cbrgb_4_pred.csv'
        fun.multi_photo_model(csv_file,4,43,[0,1,3,6],3,15,12)
    if tn==8:
        csv_file='171112_NewApp_Training_Set_pred.csv'
        #for tup in [[1,0],[2,20],[2,21],[3,30],[3,31],[3,32],[3,33],[4,40],[4,41],[4,42],[4,43],[4,4]]:
        for tup in [[2,0]]:
            result=fun.multi_photo_model(csv_file,tup[0],tup[1],[2,3],3,6,4)
            print result
    if tn==81:
        csv_file='171112_NewApp_Training_Set_pred.csv'
        #for tup in [[1,0],[2,20],[2,21],[3,30],[3,31],[3,32],[3,33],[4,40],[4,41],[4,42],[4,43],[4,4]]:
        for tup1 in range(1,6):
            for tup2 in range(0,6):
                result=fun.multi_photo_model(csv_file,tup1,tup2,[2,3],3,6,4)
                print result
    if tn==82:
        csv_list=['M1-T','M2-T','M5-T','M7-T','M1-L','M2-L','M5-L','M7-L','M1','M2','M5','M7']
        csv_list=['M2','M5','M7']
        csv_file='171112_NewApp_Training_Set_pred.csv'
        #for tup in [[1,0],[2,20],[2,21],[3,30],[3,31],[3,32],[3,33],[4,40],[4,41],[4,42],[4,43],[4,4]]:
        for csv_file in csv_list:
            print csv_file
            csv_file_tot="NewAppPredSet/" + csv_file + '.csv'
            for tup1 in range(1,5):
                for tup2 in range(1,5):
                    result=fun.multi_photo_model(csv_file_tot,tup1,tup2,[2,3],3,6,4)
                    print result
    if tn==10:
        csv_list=['M2','M2hc','M5','M7']
        #for tup in [[1,0],[2,20],[2,21],[3,30],[3,31],[3,32],[3,33],[4,40],[4,41],[4,42],[4,43],[4,4]]:
        for csv_file in csv_list:
            print csv_file
            csv_file_tot="SingleThreadPred/" + csv_file + '.csv'
            for tup1 in range(1,4):
                for tup2 in range(1,4):
                    result=fun.multi_photo_model(csv_file_tot,tup1,tup2,[2,3],3,6,4)
                    print result
if(0): #EXIF for single image
    image_path='G://Google Drive//Original Images//Embroidery//171113 NewApp Training Day3//IMG_8113.jpg'
    fun.print_exif_UC(image_path)
if(0): #EXIF data for all files in tree
    files = []
    start_dir = 'G://Google Drive//Original Images//Embroidery//171111 New App Photo//'
    #start_dir='G://Google Drive//Original Images//Embroidery//171114 Trifecta_patch_9_full_set//'
    start_dir='G://Google Drive//Original Images//Embroidery//171114 Single Thread Test//'
    start_dir='G://Google Drive//Original Images//Embroidery//171114 Trifecta_patch_9_full_set//'
    start_dir='G://Google Drive//Original Images//Embroidery//171115 Trifecta_patch_9_full_set//'
    start_dir='C://Users//gttho//Offline//Arylla Temp Photos//New Wash-20171115T223049Z-001//New Wash//'
    start_dir='G://Google Drive//Original Images//Embroidery//171115 Trifecta_patch_9_full_set//2.0'
    start_dir='G://Google Drive//Original Images//Embroidery//171122 TriPhoto Test'
    start_dir='G://Google Drive//Original Images//Embroidery//_Complete Photo Sets//Mi.50_v1app'
    start_dir='G://Google Drive//Original Images//CanadaGoose//171216 - Perry Manual Organization//Blank'
    start_dir='G://Google Drive//Original Images//CanadaGoose//171217 - StressTest'
    start_dir='G://Google Drive//Original Images//Embroidery//180116 New_Model_Prelim'
    start_dir='G://Google Drive//Original Images//Embroidery//180117 App35 Test'
    start_dir='G://Google Drive//Color_regression//images//v2app//Mi.50'
    start_dir="G://Google Drive//Original Images//Pumped Kicks//180207 Algo_Test//75ms"
    start_dir="C://Users//gttho//Resilio Sync//Original Images//CanadaGoose//180325-180326 DC_Wash_Conc_Sweep"

    pattern   = "*.jpg"

    for dir,_,_ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir,pattern))) 
    for file in files:
        print file,
        try:
            UC=fun.print_exif_UC(file)
            UC_new=UC.replace(':',',')
            file=file.replace('Blank//Logo',',11,')
            file=file.replace('Blank//Text',',12,')
            file=file.replace('M1//Logo',',13,')
            file=file.replace('M1//Text',',14,')
            file=file.replace('M2//Logo',',15,')
            file=file.replace('M2//Text',',16,')
            file=file.replace('M5//Logo',',17,')
            file=file.replace('M5//Text',',18,')
            file=file.replace('M7//Logo',',19,')
            file=file.replace('M7//Text',',20,')
            file=file.replace('M1hc//Logo',',21,')
            file=file.replace('M1hc//Text',',22,')
            file=file.replace('M2hc//Logo',',23,')
            file=file.replace('M2hc//Text',',24,')


       

            print UC_new
        except:
            print 'Failed to print: '+file
if(0): #EXIF data for not exclusively User_comment
    files = []
    start_dir_list=["C://Users//gttho//Resilio Sync//Original Images//Kings Gift//180409 TW Blank Copies"]

    pattern   = "*.jpg"

    for start_dir in start_dir_list:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob(os.path.join(dir,pattern))) 
    for file in files:
        print file,
        try:
            UC=fun.print_exif_UC(file)
            UC_new=UC.replace(':',',')
            file=file.replace('Blank//Logo',',11,')
            file=file.replace('Blank//Text',',12,')
            file=file.replace('M1//Logo',',13,')
            file=file.replace('M1//Text',',14,')
            file=file.replace('M2//Logo',',15,')
            file=file.replace('M2//Text',',16,')
            file=file.replace('M5//Logo',',17,')
            file=file.replace('M5//Text',',18,')
            file=file.replace('M7//Logo',',19,')
            file=file.replace('M7//Text',',20,')
            file=file.replace('M1hc//Logo',',21,')
            file=file.replace('M1hc//Text',',22,')
            file=file.replace('M2hc//Logo',',23,')
            file=file.replace('M2hc//Text',',24,')


       

            print UC_new
        except:
            print 'Failed to print: '+file
if(0): #Logistic Regression
    tn=0;
    if tn==0: #NO TRAIN
        directory='C://Users//gttho//Resilio Sync//Developer//csvExports//180321 JB Exports//180428//'
        csv_file=directory+'RGBGV Log Regression.csv'
        #csv_file=directory+'PrelimLogRegression.csv'
        df=pd.read_csv(csv_file,header=0,error_bad_lines=False,warn_bad_lines=False)

        #kill nan values
        df=fun.dropNAN(df,0)

        col_list=[['blue_avg','brightness'],['red_avg','brightness'],['green_avg','brightness'],['value_avg','brightness'],['grey_avg','brightness'],['blue_avg','red_avg','brightness'],['blue_avg','green_avg','brightness'],['blue_avg','value_avg','brightness'],['blue_avg','grey_avg','brightness'],['red_avg','green_avg','brightness'],['red_avg','value_avg','brightness'],['red_avg','grey_avg','brightness'],['green_avg','value_avg','brightness'],['green_avg','grey_avg','brightness'],['value_avg','grey_avg','brightness'],['blue_avg','red_avg','green_avg','brightness'],['blue_avg','red_avg','value_avg','brightness'],['blue_avg','red_avg','grey_avg','brightness'],['blue_avg','green_avg','value_avg','brightness'],['blue_avg','green_avg','grey_avg','brightness'],['blue_avg','value_avg','grey_avg','brightness'],['red_avg','green_avg','value_avg','brightness'],['red_avg','green_avg','grey_avg','brightness'],['red_avg','value_avg','grey_avg','brightness'],['green_avg','value_avg','grey_avg','brightness'],['blue_avg','red_avg','green_avg','value_avg','brightness'],['blue_avg','red_avg','green_avg','grey_avg','brightness'],['blue_avg','red_avg','value_avg','grey_avg','brightness'],['blue_avg','green_avg','value_avg','grey_avg','brightness'],['red_avg','green_avg','value_avg','grey_avg','brightness'],['blue_avg','red_avg','green_avg','value_avg','grey_avg','brightness']]
        for coeff_cols in col_list:
        #if True:
            #Separate out x values
            #coeff_cols=['avg',  'luxpred', 'brightness']
            train_x=df.filter(coeff_cols,axis=1)
            #print train_x
            #Separate out y values
            train_y=df.filter(['mark'],axis=1)

            result=fun.sm_logistic_regression_model(train_x,train_y)
            print coeff_cols,
            print ";",
            print result
            #result=fun.logistic_regression_model(x_train,y_train,printer)

    if tn==1: #NO TRAIN
        directory='G://Developer//csvExports//'
        csv_file=directory+'171213 CG Data Prelim.csv'
        x_cols=[790,791,770,807]
        row_keep=[[0]]
        for x_cols in [[0,3],[1,3],[2,3]]:
            printer = False
            x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
            #print x_train
            #print y_train
            result=fun.logistic_regression_model(x_train,y_train,printer,False)
                    
            print result
                    #result=fun.logistic_regression_model(x_train,y_train,printer)
    csv_export='python_export_data.csv'
    #os.remove(csv_export)
    if tn==11: #basic analysis
        directory='G://Google Drive//Datasets//'
        csv_file=directory+'171130_TriPhotoTest_NM1_NM2_AP_M_R_train.csv'
        csv_file_test=directory+'171130_TriPhotoTest_NM1_NM2_AP_M_R_test.csv'
        x_cols=[753,777,778,807]
        x_cols=[173,439,450,807]
        x_cols=[790,791,770,807]
        row_keep=[['PhotoSet',[101,103]]]
        #row_keep=[[0]]
        printer = True
        x_train,y_train,bad,stuff=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
        x_test,y_test,bad,stuff=fun.logistic_regression_prep(csv_file_test,x_cols,row_keep)
        print x_train
        print y_train
        print x_test
        print y_test
        result=fun.sm_logistic_regression_model(x_train,y_train,False,True,x_test,y_test)
        print result
        result=fun.logistic_regression_model(x_train,y_train,printer)
    if tn==110: #basic analysis
        directory='G://Google Drive//Datasets//'
        csv_file=directory+'171130_TriPhotoTest_NM1_NM2_AP_M_R_train.csv'
        csv_file=directory+'171207_WbExpTest_NM1_NM2_P_M_R_train.csv'
        #csv_file=directory+'171207_WbExpTest_NM1_NM2_P_M_R.csv'
        csv_file_test=directory+'171130_TriPhotoTest_NM1_NM2_AP_M_R_test.csv'
        csv_file_test=directory+'171207_WbExpTest_NM1_NM2_P_M_R_test.csv'
        x_cols=[753,777,778,807]
        x_cols=[173,439,450,807]
        x_cols=[790,791,770,807]
        row_keep_vec1=[['PhotoSet',[101,103]],['PhotoSet',[102,104]]]
        #row_keep_vec1=[['PhotoSet',[102,104]]]
        row_keep_vec2=[[['ExpBias',[-1.3]],['WB_Lock',[1]]],[['ExpBias',[-6]],['WB_Lock',[0]]],[['ExpBias',[-6]],['WB_Lock',[1]]]]
        for row_keep1 in row_keep_vec1:
            for row_keep2 in row_keep_vec2:
                for cr_ring in [0,1,2,3,4]:
                    for cb_ring in [0,1,2,3,4]:
                        for hue_ring in [0]:
                            #cr_ring=0;
                            #cb_ring=0;
                            #hue_ring=4;
                            x_cols=[584+cr_ring,595+cb_ring,807]
                            x_cols=[62+cr_ring,67+cb_ring,105]
                            print [cr_ring,cb_ring,hue_ring],
                            #row_keep=[['PhotoSet',[101,103]]]
                            row_keep3=copy.copy(row_keep2)
                            row_keep3.append(row_keep1)
                            row_keep=row_keep3
                            print row_keep,
                            #row_keep=[['PhotoSet',[102,104]]]
                            #row_keep=[[0]]
                            printer = False
                            x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
                            x_test,y_test,bad,stuff=fun.logistic_regression_prep(csv_file_test,x_cols,row_keep)
                            #print x_train
                            #print y_train
                            #print x_test
                            #print y_test
                            result=fun.sm_logistic_regression_model(x_train,y_train,printer,True,x_test,y_test)
                    
                    #print result
                    #result=fun.logistic_regression_model(x_train,y_train,printer)
    if tn==13: #analysis with different x_col
        csv_file='171120_Flash_noFlash_Logit_Analysis.csv'
        x_cols_list=[[1,2,3,4,5,6,7,8,9,10,11,12,13,32],[15,16,17,18,19,20,21,22,23,24,25,26,27,28,32],[1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,32],[2,3,6,32],[17,18,21,32],[2,3,6,17,18,21,32]]
        row_keep=[['All-Distance',[5]],['All-Conc',[0,15]]]
        printer = False
        for x_cols in x_cols_list:
            print x_cols
            x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
            result=fun.logistic_regression_model(x_train,y_train,printer)
            print result[0]
            print " "
    if tn==10: #analysis with varying row keeps
        csv_file='171114_Single_Thread.csv'
        x_cols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]
        
        printer = False
        val_keep_list_list=[[0,2],[0,3],[0,4],[0,5],[0,6]];
        for row_keep_vec in val_keep_list_list:
            row_keep=[['PhotoSet',row_keep_vec],['Count',[0]]]
            x_train,y_train=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
            result=fun.logistic_regression_model(x_train,y_train)
            print row_keep_vec
            print result
    if tn==12:
        csv_file='171114 Trifecta_patch_9_full_set.csv'
        x_cols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]
        row_keep=[['PhotoSet',[7]],['Count',[0]]]
        printer = False
        x_train,y_train=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
        result=fun.logistic_regression_model(x_train,y_train)
    if tn==111:
        csv_file='171114 Trifecta_patch_9_full_set.csv'
        x_cols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]
        row_keep=[['PhotoSet',[0]],[' Count',[0]]]
        coeffs=[0.0436167606411, 8.6996117838e-10, -0.920589433307, -0.0112127921241, 0.00384585550982, -0.0200985548372, -0.717652120036, -0.122021900781, 0.556816379319, 0.167241487045, 0.395435506365, -0.0171866536177, -0.161380872953, 0.000309535616489, 0.00105015226234]
        intercept=-0.02540291
        printer = False
        x_train,y_train=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
        result=fun.logistic_model_tester(x_train,y_train,coeffs,intercept)
    if tn==112:
        csv_file='171115 Trifecta_patch_9_treated_set.csv'
        x_cols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]
        val_keep_list_list=[[1,2],[3,4],[5,6],[7,8]];
        val_keep_list_list=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]];
        for keep_list in val_keep_list_list:
            row_keep=[['PhotoSet',keep_list]]
            coeffs=[0.0436167606411, 8.6996117838e-10, -0.920589433307, -0.0112127921241, 0.00384585550982, -0.0200985548372, -0.717652120036, -0.122021900781, 0.556816379319, 0.167241487045, 0.395435506365, -0.0171866536177, -0.161380872953, 0.000309535616489, 0.00105015226234]
            intercept=-0.02540291
            printer = False
            x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
            result=fun.logistic_model_tester(x_train,y_train,coeffs,intercept)
            print keep_list,
            print result
    if tn==81:
        csv_file='171115_NewApp_FixedSet.csv'
        x_cols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]
        #x_cols=[1,2,3,4,7,14,15,17]
        #x_cols=[7,14,15,17]
        #val_keep_list_list=[[11,13],[12,14],[11,15],[12,16],[11,17],[12,18],[11,19],[12,20],[11,21],[12,22],[11,23],[12,24],[11,12,13,14],[11,12,15,16],[11,12,17,18],[11,12,19,20],[11,12,21,22],[11,12,23,24]]
        #val_keep_list_list=[[11,17],[12,18],[11,21],[12,22]]
        val_keep_list_list=[[11,21]]
        coeffs_list=[]
        intercepts_list=[]
        for keep_list in val_keep_list_list:
            row_keep=[['PhotoSet',keep_list]]
            print row_keep
            for perc in np.linspace(0.25,.25,1):
                x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep,perc)
                result=fun.logistic_regression_model(x_train,y_train,tester_switch=True,xTst=x_test,yTst=y_test)
                print str(perc)+";"+str(result[0])+";"+str(result[1])+";"+str(result[2])
                coeffs_list.append(result[3][0])
                intercepts_list.append(result[4][0])
            x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
            break
            accuracy_holder_plus=[]
            for index in range(0,len(coeffs_list)):
                
                result2=fun.logistic_model_tester(x_train,y_train,coeffs_list[index],intercepts_list[index])
                accuracy_holder_plus.append(result2[7])
            acc_list_mat=np.array(accuracy_holder_plus)
            acc_mat=np.transpose(acc_list_mat)
            fixed_average_holder=[]
            average_holder=[]
            for i in range(acc_mat.shape[0]):
                if np.average(acc_mat[i,:])>0.5:
                    fixed_average_holder.append(1)
                else:
                    fixed_average_holder.append(0)
            for i in range(acc_mat.shape[1]):
                average_holder.append(np.average(acc_mat[:,i]))

           # print fixed_average_holder
            print np.average(fixed_average_holder)
            print np.max(average_holder)
            print np.average(average_holder)
            print np.min(average_holder)

                #for val in result2:
                #    print val,
                #    print ";",
                #print " "
                    #print row_keep
                    #print str(perc)+";"+str(result[0])+";"+str(result[1])+";"+str(result[2])
                    #print perc,
                    #print ";",
                    #print result[4][0],
                    #print ";",
                    #for val in result[3]:
                    #    for val2 in val:
                    #        print str(val2)+";",
                    #print " "
    if tn==82: #compare training pics with testing pics using sm
        csv_file='171115_NewApp_FixedSet.csv'
        csv_file_tester='171122_AppAccuracy.csv'
        csv_file='171122_CP_Sweep_M1ch-L_train.csv'
        csv_file_tester='171122_CP_Sweep_M1ch-L_test.csv'
        csv_file='171123_TriPhotoTest_train.csv'
        csv_file_tester='171123_TriPhotoTest_test.csv'
        csv_file='171123_DistanceBigTest_train.csv'
        csv_file_tester='171123_DistanceBigTest_test.csv'
        x_cols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]
        data_cols=[2,3,4,5,6,7,14,15]
        data_cols=[9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
        data_cols=[8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76]
        x_cols=[7,17]
        x_cols=[7,14,15,17]
        x_cols_list=[[3,14,15,17],[4,14,15,17],[5,14,15,17],[6,14,15,17],[7,14,15,17]]
        x_cols_list=[[4,7,14,15,17]]
        #x_cols=[1,2,3,4,7,14,15,17]
        #x_cols=[7,14,15,17]
        #val_keep_list_list=[[11,13],[12,14],[11,15],[12,16],[11,17],[12,18],[11,19],[12,20],[11,21],[12,22],[11,23],[12,24],[11,12,13,14],[11,12,15,16],[11,12,17,18],[11,12,19,20],[11,12,21,22],[11,12,23,24]]
        #val_keep_list_list=[[11,17],[12,18],[11,21],[12,22]]
        #val_keep_list_list=[[11,21],[12,22],[11,23],[12,24]]
        val_keep_list_list=[[11,21]]
        keep_list=[11,21]
        keep_list=[101,103]
        #for keep_list in val_keep_list_list:
        for iter in [1]:
            print iter
            for val in it.combinations(data_cols,iter):
            #for x_cols in x_cols_list:
                x_cols=list(val)
                #x_cols.append(56)
                #x_cols.append(17)
                #x_cols.append(18)
                x_cols.append(78)
                x_cols=sorted(x_cols)
                row_keep=[['PhotoSet',keep_list]]
                x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep)
                x_train2,y_train2,x_test2,y_test2=fun.logistic_regression_prep(csv_file_tester,x_cols,row_keep)
                try:
                    result=fun.sm_logistic_regression_model(x_train,y_train,printer=False,xTst=x_train2,yTst=y_train2)
                    result.extend(x_cols)
                    with open(r'python_export_data_'+str(iter)+'.csv', 'ab') as f:
                        writer = csv.writer(f)
                        writer.writerow(result)
                except:
                    print "Failed x_col: ",
                    print x_cols
                #print str(result[0])+";"+str(result[1])+";"+str(result[2])
                #print result[4][0],
                #print ";",
                #for val in result[3]:
                #    for val2 in val:
                #        print str(val2)+";",
    if tn==821: #compare training pics with testing pics using sm
        directory='G://Google Drive//Datasets//'
        csv_file='SuperDuperSet_train.csv'
        csv_file=directory+'171210_ExpWb_InitialPhotoset_rings.csv'
        dataframe_in_train=pd.read_csv(csv_file,header=0)
        csv_file='SuperDuperSet_test.csv'
        csv_file=directory+'171210_ExpWb_InitialPhotoset_rings.csv'
        dataframe_in_test=pd.read_csv(csv_file,header=0)
        data_cols=[38,258,423,478,533,643,698,808,863,918,973,1083,2458,2523,2578,2798,2963,3018,3073,3183,3238,3348,3403,3458,3513,3623]
        data_cols=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
        row_keep=[[0]]
        #for keep_list in val_keep_list_list:
        #for iter in range(10):
        for row_keep in [['PhotoSet',[101,103]],['PhotoSet',[102,104]],['PhotoSet',[101,105]],['PhotoSet',[102,106]],['PhotoSet',[101,107]],['PhotoSet',[102,108]],['PhotoSet',[101,109]],['PhotoSet',[102,110]]]:
            iter=1
            row_keep=[row_keep]
            for val in it.combinations(data_cols,iter):
            #for x_cols in x_cols_list:
                dataframe_in_test_copy=copy.copy(dataframe_in_test)
                dataframe_in_train_copy=copy.copy(dataframe_in_train)
                x_cols=list(val)
                #x_cols.append(4963)
                x_cols.append(55)
                x_cols=sorted(x_cols)
                #row_keep=[[0]]
                
                x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep,dataframe_checker=True,dataframe_input=dataframe_in_train_copy)
                #x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep,dataframe_checker=True,dataframe_input=dataframe_in_train_copy)
                #x_train2,y_train2,x_test2,y_test2=fun.logistic_regression_prep(csv_file,x_cols,row_keep,dataframe_checker=True,dataframe_input=dataframe_in_test_copy)
                #print "data_done"
                try:
                #if True:
                    result=fun.sm_logistic_regression_model(x_train,y_train)
                    print row_keep,
                    print ";",
                    print x_cols,
                    print ";",
                    print result
                    #print "anal_done"
                    #result.extend(x_cols)
                    #with open(r'python_export_data_'+str(iter)+'superData.csv', 'ab') as f:
                    #    writer = csv.writer(f)
                    #    writer.writerow(result)
                except:
                    print "Failed x_col: ",
                    print x_cols
                #print str(result[0])+";"+str(result[1])+";"+str(result[2])
                #print result[4][0],
                #print ";",
                #for val in result[3]:
                #    for val2 in val:
                #        print str(val2)+";",
    if tn==83:
        csv_file='171115_NewApp_FixedSet.csv'
        data_cols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        #data_cols=[1,3,4,7,14]
        y_col=17
        #val_keep_list_list=[[11,13],[12,14],[11,15],[12,16],[11,17],[12,18],[11,19],[12,20],[11,21],[12,22],[11,23],[12,24],[11,12,13,14],[11,12,15,16],[11,12,17,18],[11,12,19,20],[11,12,21,22],[11,12,23,24]]
        #val_keep_list_list=[[11,17],[12,18],[11,21],[12,22]]
        val_keep_list_list=[[11,17]]
        val_keep_list_list=[[11,12,21,22]]
        val_keep_list_list=[[11,21]]
        #for iter in range(2,len(data_cols)+1):
        #for iter in [1,14,15]:
        #for iter in [12,13]:
        for iter in [15]:
           for val in it.combinations(data_cols,iter):
            #x_cols=[1,2,3,4,7,14,15,17]
                x_cols=list(val)
                x_cols.append(y_col)
                for keep_list in val_keep_list_list:
                    row_keep=[['PhotoSet',keep_list]]
                    print row_keep,
                    print ";",
                    print x_cols,
                    print ";",
                    x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep,0.66)
                    result=fun.logistic_regression_model(x_train,y_train,printer=False)
                    print ";",
                    print str(result[0])+";"+str(result[1])+";"+str(result[2])
                    print ";"
    if tn==84:
        csv_file='171117_Distance_Test.csv'
        x_cols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]
        #x_cols=[1,2,3,4,7,14,15,17]
        #x_cols=[7,14,15,17]
        #val_keep_list_list=[[11,13],[12,14],[11,15],[12,16],[11,17],[12,18],[11,19],[12,20],[11,21],[12,22],[11,23],[12,24],[11,12,13,14],[11,12,15,16],[11,12,17,18],[11,12,19,20],[11,12,21,22],[11,12,23,24]]
        #val_keep_list_list=[[11,17],[12,18],[11,21],[12,22]]
        val_keep_list_list=[[11,17]]
        val_keep_list=[[0]]
        x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols)
        result=fun.logistic_regression_model(x_train,y_train,printer=False)
        print str(result[0])+";"+str(result[1])+";"+str(result[2])
if(0): #CanadaGoose Logistic Regression
    #0 - logistic regression
    #1 - threshold modeling
    #2 - threshold model testing
    #3 - mean finder
    #4 - threshold modeling v2
    #5 - threshold modeling v2 with filtered Blanks and Detailed Report
    #6 - threshold modeling v3 - arbitrary ROI_max and ROI_count
    test_index=6

    line_by_line_check=0

    files=[]
    start_dir='csv_import'
    pattern   = "*.csv"
    print 'prog1'
    for dir,_,_ in os.walk(start_dir): 
        files.extend(glob(os.path.join(dir,pattern))) 
    for file in files:
        #print 'prog2'
        filename=file.rpartition("\\")[2]
        #Import csv to dataframe only once

        if (line_by_line_check):
            print filename

        #grab ring number from filename
        f_list=filename.split("_")
        end_bit=f_list[-1]
        ring_split=end_bit.split("'")
        ring_string=ring_split[-1]
        ring_string=ring_string[:-4]
        #ring=55
        try:
            ring=int(ring_string)-500
        except:
            ring=55
        #print ring
        ring=36

        df=pd.read_csv(file,header=0,error_bad_lines=False,warn_bad_lines=False)
        
        if (0): #blank concat
            df_b=pd.read_csv("G://Developer//csvExports//180208 PK Exports//StandardBlanks//ColoredBack_2Dbaseline_12Ring.csv",error_bad_lines=False,warn_bad_lines=False)
            #print df_b
            df=df.append(df_b);

      #  df=fun.set_column_sequence(df,['path','formulation','ap','shoe','brand','location','size','colour','lux','datetime'])
       # print df

       # df=fun.path_filter(df,"Solid_Square_80mS_100mS")

        t0 = time.time()
        

        #fun.pk_modeler(df,ring)

        t1 = time.time()

        total = t1-t0

        #print "Time: ",
        #print total

        ##remove NAN values from df
        #column_where_numbers_start=8;
        #header_list=df.columns.values.tolist()
        ##FIRST, CONVERT COLUMNS TO FLOATS
        #for index in range(column_where_numbers_start,len(header_list)): #ONLY OPERATE ON COLUMNS PAST COLUMN 5
        #    df[header_list[index]] = df[header_list[index]].apply(pd.to_numeric,errors='coerce')
        ##THEN REMOVE NAN VALUES
        #df=df.dropna(subset = header_list[column_where_numbers_start:])

        #replace '-nan(ind)' with 0

        #df=df.replace('-nan(ind)',0)

        if line_by_line_check:
            print "ORIGINAL DATAFRAME: "
            print df

        df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if line_by_line_check:
            print "DROP UNNAMED: "
            print df

        #df=fun.dropNAN(df,35)
        if line_by_line_check:
            print "DROP NAN VALUES: "
            print df
        #print df

        df=df.loc[df['guess'].isin([0,1])]

        #Make Dataframe of Blanks
        #df_blank=df.loc[df['ink'].str.contains('blank|Blank')]
        df_blank=df.loc[df['mark']==0]
        
        #df_blank=df.loc[df['Path'].str.contains('blank|Blank')]

        if line_by_line_check:
            print "BLANK DATAFRAME"
            print df_blank


        ##############_PATH FILTERS_#####################
        path_list=[]
       # path_list.append('180221 Tag_Variations_80mS_AP1')
       # path_list.append('Yeezy')

     #   path_list.append('180228')
      #  path_list.append('180301').
        #path_list.append('180327 New_Exp_Bias_-1.6')
        #path_list.append('180327 DC_Wash_Different_Inks')
       # path_list.append('180325-180326 DC_Wash_Conc_Sweep')
     #   path_list.append('180409 White_Tag_PVP_Ink')
        path_list.append('skip')
    #    path_list.append("Solid_Square_80mS_100mS")

      #  path_list=["skip"]

        ###############_FORMULATOINS_####################
        formulation_superlist=[]

        formulation_superlist.append([["skip"],[['skip','skip','skip']]])

        #[FORMULATION,RECIPE]->[FORMULATION,[INK,BINDER,SOLVENT]]->[FORMULATION,[[INK,BINDER,SOLVENT],[INK2,BINDER2,SOLVENT2]]]

        mod_list=[]

        mod_list.append('skip')


        ROI_list=range(3)
        #x_col_list=[] #these should be COLUMN HEADERS
        #x_col_list.append()

        ROI_max=10;
        ROI_count=1;

        for path in path_list:
            for formulation_list in formulation_superlist: #elements with the same formulation are clustered together
                for recipe in formulation_list[1]: #elements with different recipes are run
                    for mod in mod_list:
                        #print "running loops..."
                        #print test_index
                        ROI_result=[]
                        ROI_avg_result=[]
                        try:
                        #if True:
                            if test_index==0:
                                for ROI in ROI_list:
                                    df_input=copy.copy(df)
                                    df_blank_input=copy.copy(df_blank)
                                    formulation=formulation_list[0][0]
                                    df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                                    df_fin=fun.cg_combine_print_blank(df_input,df_blank)
                                    if line_by_line_check:
                                        print df_fin
                                    df_fin.head()
                                    [x_train,y_train]=fun.logistic_regression_prep_cg(df_input,df_blank_input,ROI,ROI_max)
                                    ROI_avg_result.append(y_train.Mark.mean())
                                    if test_index==0:
                                        result=fun.logistic_regression_model(x_train,y_train)
                                
                                    ROI_result.append(result[0])
                                print filename,
                                print ";",
                                print formulation_list[0][0],
                                print ";",
                                print recipe[0],
                                print ";",
                                print recipe[1],
                                print ";",
                                print recipe[2],
                                print ";",
                                print mod,
                                print ";",
                                print ROI_result[0],
                                print ";",
                                print ROI_result[1],
                                print ";",
                                print ROI_result[2]
                            if test_index==1 or test_index==4 or test_index==5 or test_index==6:
                                df_input=copy.copy(df)
                                df_blank_input=copy.copy(df_blank)
                                formulation=formulation_list[0][0]

                                if line_by_line_check:
                                    print 'INITIAL BLANK: '
                                    print df_blank.head()
                                    print 'INITIAL PRINT: '
                                    print df_input.head()
                                #df_input=fun.path_filter(df_input,path)
                                #df_input=df_input.loc[df['conc']==mod]
                                df_input=fun.arbitrary_include(df_input,'conc',mod)
                                #df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)f
                                #print recipe
                                #print df_blank.iloc[:,0:5]
                                #if recipe[1]==3:
                                #    black_recipe=1
                                #elif recipe[1]==4:
                                #    blank_recipe=2
                                if test_index==5:
                                    df_blank_input=fun.cg_dataframe_filter(df_blank_input,'blank',['skip',recipe[1],'skip'],mod)
                                #df_blank_input=fun.cg_dataframe_filter(df_blank_input,'Blank',['skip',blank_recipe,'skip'],mod)
                                #print df_blank.iloc[:,0:5]
                                #df_input=fun.arbitrary_exclude(df_input,'mark','0')
                                df_input=df_input.loc[df['mark']==1]
                                df_fin=fun.cg_combine_print_blank(df_input,df_blank_input)
                                if line_by_line_check:
                                    print 'FINAL DATAFRAME'
                                    print df_fin
                                #print df_fin
                                if test_index==1:
                                    result=fun.cg_redundancy_modeler(df_fin)
                                elif test_index==4:
                                    result=fun.cg_redundancy_modeler_v2(df_fin,ROI_max)
                                elif test_index==5:
                                    result=fun.cg_redundancy_modeler_v2(df_fin,ROI_max)
                                    result2=fun.cg_redundancy_tester_detail(df_fin,result[1],result[2],result[3],scan_size=ROI_max,print_failures=False)
                                elif test_index==6:
                                    result=fun.cg_redundancy_modeler_v3(df_fin,ROI_max,ROI_count)
                                

                                print filename,
                                print ";",
                                print formulation_list[0][0],
                                print ";",
                                print str(recipe[0])+"-"+str(recipe[1])+"-"+str(recipe[2]),
                                print ";",
                                #print recipe[1],
                                #print ";",
                                #print recipe[2],
                                #print ";",
                                print mod,
                                print ";",
                                print result[0][0],
                                print ";",
                                print result[0][1],
                                print ";",
                                print result[0][2],
                                print ";",
                                print result[0][3],
                                print ";",
                                print result[0][4],
                                print ";",
                                print result[1],
                                print ";",
                                print result[2],
                                print ";",
                                print result[3],
                                print ";",
                                print result[4]
                            if test_index==2:
                                df_input=copy.copy(df)
                                df_blank_input=copy.copy(df_blank)
                                formulation=formulation_list[0][0]
                               # df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                                #df_fin=fun.cg_combine_print_blank(df_input,df_blank)

                                df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                                #print recipe
                                #print df_blank.iloc[:,0:5]
                            
                                df_blank_input=fun.cg_dataframe_filter(df_blank_input,'blank',['skip',recipe[1],'skip'],mod)
                            
                                #print df_blank.iloc[:,0:5]
                                df_fin=fun.cg_combine_print_blank(df_input,df_blank_input)

                                #print df_fin 

                                #SFM-237
                                best_roi_thresh=[0.189261947,0.189261947,0.304590368]
                                best_dec_thresh=[0.65,0.75,0.65]
                                best_redundancy=1.5

                                #BR-237
                                best_roi_thresh=[0.337868789,0.602583263,0.426106947]
                                best_dec_thresh=[0.65,0.65,0.65]
                                best_redundancy=0.5

                                 #BR-2347
                                best_roi_thresh=[0.44042734,0.651658701,0.68196173152578421]
                                best_dec_thresh=[0.55,0.55,0.65]
                                best_redundancy=0.5

                                #BR-2347
                                best_roi_thresh=[0.4564061669324827, 0.65470494417862846, 0.56751727804359386]
                                best_dec_thresh=[0.55,0.65,0.65]
                                best_redundancy=0.5

                                #BR-2347-fin
                                best_roi_thresh=[0.419723551,0.669856459,0.57761828814] 
                                best_dec_thresh=[0.75,0.65,0.75]
                                best_redundancy=0.5

                                #BSB-1_2-17_EVm1_Stress
                                best_roi_thresh=[1.219298246,1.368421053,0.461456672] 
                                best_dec_thresh=[0.5,0.5,0.5]
                                best_redundancy=0.5

                                 #BSB-2_2-17_EVm1_Stress
                                best_roi_thresh=[0.878256247,1.883838384,0.884635832] 
                                best_dec_thresh=[0.5,0.5,0.5]
                                best_redundancy=1.5

                                super_coeff_dict={"csv_import\KM_Black_StressTest_180125_2.csv , Ink , skip-1-skip , BSB":[0.903508772,1.526315789,0.384370016,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_2.csv , Ink , skip-2-skip , BSB":[0.451355662,1.485911749,0.589048379,0.5,0.5,0.5,1.5],"csv_import\KM_Black_StressTest_180125_2.csv , Ink , skip-1-skip , BWS":[0.792397661,0.863104732,1.763157895,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_2.csv , Ink , skip-2-skip , BWS":[0.231525784,1.432748538,2.075757576,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_11.csv , Ink , skip-1-skip , BSB":[0.815789474,2,1.921052632,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11.csv , Ink , skip-2-skip , BSB":[2,2.01010101,1.800372142,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11.csv , Ink , skip-1-skip , BWS":[2,2,2.212121212,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11.csv , Ink , skip-2-skip , BWS":[2.080808081,2.166666667,2.191919192,0.5,0.5,0.5,1.5],"csv_import\KM_Black_StressTest_180125_17.csv , Ink , skip-1-skip , BSB":[1.791600213,2,1.418926103,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_17.csv , Ink , skip-2-skip , BSB":[1.924242424,1.97979798,0.619351409,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_17.csv , Ink , skip-1-skip , BWS":[1.255980861,1.729665072,1.704412547,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_17.csv , Ink , skip-2-skip , BWS":[1.917862839,2.005050505,0.782296651,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_11-17-.csv , Ink , skip-1-skip , BSB":[0.5,2,2,0.45,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-17-.csv , Ink , skip-2-skip , BSB":[1.921052632,2,2.141414141,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-17-.csv , Ink , skip-1-skip , BWS":[2.161616162,1.862307283,2.095959596,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-17-.csv , Ink , skip-2-skip , BWS":[2.075757576,2.121212121,2.176767677,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-2-.csv , Ink , skip-1-skip , BSB":[1.368421053,1.921052632,2,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-2-.csv , Ink , skip-2-skip , BSB":[2,2,2.166666667,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-2-.csv , Ink , skip-1-skip , BWS":[2,2.050505051,2.116161616,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-2-.csv , Ink , skip-2-skip , BWS":[2.171717172,2.126262626,2,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-2-17-.csv , Ink , skip-1-skip , BSB":[2,2,2,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-2-17-.csv , Ink , skip-2-skip , BSB":[1.921052632,2,2.191919192,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-2-17-.csv , Ink , skip-1-skip , BWS":[2.176767677,1.968367889,2.186868687,0.5,0.5,0.5,2.5],"csv_import\KM_Black_StressTest_180125_11-2-17-.csv , Ink , skip-2-skip , BWS":[2.141414141,2,0.421052632,0.5,0.5,0.45,2.5],"csv_import\KM_Black_StressTest_180125_2-17-.csv , Ink , skip-1-skip , BSB":[1.219298246,1.368421053,0.461456672,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_2-17-.csv , Ink , skip-2-skip , BSB":[0.878256247,1.883838384,0.884635832,0.5,0.5,0.5,1.5],"csv_import\KM_Black_StressTest_180125_2-17-.csv , Ink , skip-1-skip , BWS":[1.098086124,0.868155237,0.74056353,0.5,0.5,0.5,0.5],"csv_import\KM_Black_StressTest_180125_2-17-.csv , Ink , skip-2-skip , BWS":[0.649654439,1.830675173,2.070707071,0.5,0.5,0.5,0.5]}
                                super_coeff_dict={"csv_import\KM_Black_StressTest_180125_0.csv , Ink , skip-1-skip , BSB":[0.618022329,1.684210526,1.368421053,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_0.csv , Ink , skip-2-skip , BSB":[1.12333865,1.139819245,0.074960128,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_0.csv , Ink , skip-1-skip , BWS":[0.421052632,1.210526316,1.052631579,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_0.csv , Ink , skip-2-skip , BWS":[1.087985114,2,0.242955875,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_1.csv , Ink , skip-1-skip , BSB":[0.618022329,1.684210526,1.368421053,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_1.csv , Ink , skip-2-skip , BSB":[1.12333865,1.139819245,0.074960128,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_1.csv , Ink , skip-1-skip , BWS":[0.421052632,1.210526316,1.052631579,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_1.csv , Ink , skip-2-skip , BWS":[1.087985114,2,0.242955875,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_2.csv , Ink , skip-1-skip , BSB":[0.903508772,1.526315789,0.384370016,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_2.csv , Ink , skip-2-skip , BSB":[0.451355662,1.485911749,0.589048379,0.5,0.5,0.5,1.5],
    "csv_import\KM_Black_StressTest_180125_2.csv , Ink , skip-1-skip , BWS":[0.792397661,0.863104732,1.763157895,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_2.csv , Ink , skip-2-skip , BWS":[0.231525784,1.432748538,2.075757576,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_3.csv , Ink , skip-1-skip , BSB":[0.618022329,1.684210526,1.368421053,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_3.csv , Ink , skip-2-skip , BSB":[1.12333865,1.139819245,0.074960128,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_3.csv , Ink , skip-1-skip , BWS":[0.421052632,1.210526316,1.052631579,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_3.csv , Ink , skip-2-skip , BWS":[1.087985114,2,0.242955875,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_4.csv , Ink , skip-1-skip , BSB":[0.618022329,1.684210526,1.368421053,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_4.csv , Ink , skip-2-skip , BSB":[1.12333865,1.139819245,0.074960128,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_4.csv , Ink , skip-1-skip , BWS":[0.421052632,1.210526316,1.052631579,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_4.csv , Ink , skip-2-skip , BWS":[1.087985114,2,0.242955875,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_5.csv , Ink , skip-1-skip , BSB":[0.573896863,1.526315789,0.777246146,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_5.csv , Ink , skip-2-skip , BSB":[0.664805954,0.772195641,0.135566188,0.5,0.5,0.5,1.5],
    "csv_import\KM_Black_StressTest_180125_5.csv , Ink , skip-1-skip , BWS":[0.292131845,0.812599681,0.292131845,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_5.csv , Ink , skip-2-skip , BWS":[0.609250399,0.649654439,-0.047581074,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_6.csv , Ink , skip-1-skip , BSB":[0.618022329,1.684210526,0.812599681,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_6.csv , Ink , skip-2-skip , BSB":[1.005847953,1.010898458,0.085061138,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_6.csv , Ink , skip-1-skip , BWS":[0.399521531,1.763157895,0.649654439,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_6.csv , Ink , skip-2-skip , BWS":[0.898458267,0.990696438,2.035353535,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_7.csv , Ink , skip-1-skip , BSB":[0.618022329,1.684210526,1.368421053,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_7.csv , Ink , skip-2-skip , BSB":[1.12333865,1.139819245,0.074960128,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_7.csv , Ink , skip-1-skip , BWS":[0.421052632,1.210526316,1.052631579,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_7.csv , Ink , skip-2-skip , BWS":[1.087985114,2,0.242955875,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_8.csv , Ink , skip-1-skip , BSB":[0.618022329,1.684210526,1.368421053,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_8.csv , Ink , skip-2-skip , BSB":[1.12333865,1.139819245,0.074960128,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_8.csv , Ink , skip-1-skip , BWS":[0.421052632,1.210526316,1.052631579,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_8.csv , Ink , skip-2-skip , BWS":[1.087985114,2,0.242955875,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_9.csv , Ink , skip-1-skip , BSB":[0.633173844,1.684210526,1.368421053,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_9.csv , Ink , skip-2-skip , BSB":[1.13343966,1.14486975,0.095162148,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_9.csv , Ink , skip-1-skip , BWS":[0.007974482,1.368421053,1.447368421,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_9.csv , Ink , skip-2-skip , BWS":[1.368421053,2.060606061,0.736842105,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_10.csv , Ink , skip-1-skip , BSB":[0.669856459,1.763157895,0.817650186,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_10.csv , Ink , skip-2-skip , BSB":[0.827751196,1.429027113,0.29851143,0.5,0.5,0.5,1.5],
    "csv_import\KM_Black_StressTest_180125_10.csv , Ink , skip-1-skip , BWS":[0.842902711,0.950292398,0.777246146,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_10.csv , Ink , skip-2-skip , BWS":[1.128389155,1.55661882,0.374269006,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_11.csv , Ink , skip-1-skip , BSB":[0.815789474,2,1.921052632,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11.csv , Ink , skip-2-skip , BSB":[2,2.01010101,1.800372142,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11.csv , Ink , skip-1-skip , BWS":[2,2,2.212121212,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11.csv , Ink , skip-2-skip , BWS":[2.080808081,2.166666667,2.191919192,0.5,0.5,0.5,1.5],
    "csv_import\KM_Black_StressTest_180125_12.csv , Ink , skip-1-skip , BSB":[0.105263158,1.763157895,2,0.45,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_12.csv , Ink , skip-2-skip , BSB":[2,1.931153642,2.136363636,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_12.csv , Ink , skip-1-skip , BWS":[2,2,2.212121212,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_12.csv , Ink , skip-2-skip , BWS":[2.080808081,2.166666667,2.191919192,0.5,0.5,0.5,1.5],
    "csv_import\KM_Black_StressTest_180125_13.csv , Ink , skip-1-skip , BSB":[0.618022329,1.684210526,1.368421053,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_13.csv , Ink , skip-2-skip , BSB":[1.12333865,1.139819245,0.074960128,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_13.csv , Ink , skip-1-skip , BWS":[0.421052632,1.210526316,1.052631579,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_13.csv , Ink , skip-2-skip , BWS":[1.087985114,2,0.242955875,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_14.csv , Ink , skip-1-skip , BSB":[0.894736842,1.684210526,1.447368421,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_14.csv , Ink , skip-2-skip , BSB":[1.093035619,1.13476874,0.074960128,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_14.csv , Ink , skip-1-skip , BWS":[0.421052632,1.052631579,0.424774056,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_14.csv , Ink , skip-2-skip , BWS":[1.057682084,1.763157895,0.32881446,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_15.csv , Ink , skip-1-skip , BSB":[0.70015949,1.842105263,1.526315789,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_15.csv , Ink , skip-2-skip , BSB":[2,1.336788942,0.121743753,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_15.csv , Ink , skip-1-skip , BWS":[0.578947368,2,0.736842105,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_15.csv , Ink , skip-2-skip , BWS":[1.418926103,2,0.506911217,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_16.csv , Ink , skip-1-skip , BSB":[0.654704944,1.605263158,1.447368421,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_16.csv , Ink , skip-2-skip , BSB":[1.082934609,1.099415205,0.034556087,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_16.csv , Ink , skip-1-skip , BWS":[0.686337055,1.052631579,1.052631579,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_16.csv , Ink , skip-2-skip , BWS":[1.526315789,0.950292398,0.273258905,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_17.csv , Ink , skip-1-skip , BSB":[1.791600213,2,1.418926103,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_17.csv , Ink , skip-2-skip , BSB":[1.924242424,1.97979798,0.619351409,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_17.csv , Ink , skip-1-skip , BWS":[1.255980861,1.729665072,1.704412547,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_17.csv , Ink , skip-2-skip , BWS":[1.917862839,2.005050505,0.782296651,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_18.csv , Ink , skip-1-skip , BSB":[1.921052632,2,1.763157895,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_18.csv , Ink , skip-2-skip , BSB":[1.921052632,1.921052632,0.973684211,0.5,0.5,0.5,1.5],
    "csv_import\KM_Black_StressTest_180125_18.csv , Ink , skip-1-skip , BWS":[2,1.921052632,2,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_18.csv , Ink , skip-2-skip , BWS":[2,1.842105263,2,0.5,0.5,0.5,1.5],
    "csv_import\KM_Black_StressTest_180125_19.csv , Ink , skip-1-skip , BSB":[0.184210526,0.973684211,1.842105263,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_19.csv , Ink , skip-2-skip , BSB":[1.447368421,1.921052632,1.763157895,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_19.csv , Ink , skip-1-skip , BWS":[1.842105263,1.052631579,1.842105263,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_19.csv , Ink , skip-2-skip , BWS":[1.921052632,1.052631579,1.368421053,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-17-.csv , Ink , skip-1-skip , BSB":[0.5,2,2,0.45,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-17-.csv , Ink , skip-2-skip , BSB":[1.921052632,2,2.141414141,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-17-.csv , Ink , skip-1-skip , BWS":[2.161616162,1.862307283,2.095959596,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-17-.csv , Ink , skip-2-skip , BWS":[2.075757576,2.121212121,2.176767677,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-2-.csv , Ink , skip-1-skip , BSB":[1.368421053,1.921052632,2,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-2-.csv , Ink , skip-2-skip , BSB":[2,2,2.166666667,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-2-.csv , Ink , skip-1-skip , BWS":[2,2.050505051,2.116161616,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-2-.csv , Ink , skip-2-skip , BWS":[2.171717172,2.126262626,2,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-2-17-.csv , Ink , skip-1-skip , BSB":[2,2,2,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-2-17-.csv , Ink , skip-2-skip , BSB":[1.921052632,2,2.191919192,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-2-17-.csv , Ink , skip-1-skip , BWS":[2.176767677,1.968367889,2.186868687,0.5,0.5,0.5,2.5],
    "csv_import\KM_Black_StressTest_180125_11-2-17-.csv , Ink , skip-2-skip , BWS":[2.141414141,2,0.421052632,0.5,0.5,0.45,2.5],
    "csv_import\KM_Black_StressTest_180125_2-17-.csv , Ink , skip-1-skip , BSB":[1.219298246,1.368421053,0.461456672,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_2-17-.csv , Ink , skip-2-skip , BSB":[0.878256247,1.883838384,0.884635832,0.5,0.5,0.5,1.5],
    "csv_import\KM_Black_StressTest_180125_2-17-.csv , Ink , skip-1-skip , BWS":[1.098086124,0.868155237,0.74056353,0.5,0.5,0.5,0.5],
    "csv_import\KM_Black_StressTest_180125_2-17-.csv , Ink , skip-2-skip , BWS":[0.649654439,1.830675173,2.070707071,0.5,0.5,0.5,0.5]}


                            
                            
                                #key=filename+" , "+formulation_list[0][0]+" , "+str(recipe[0])+"-"+str(recipe[1])+"-"+str(recipe[2])+" , "+mod
                                #coeff_list=super_coeff_dict[key]
                                #best_roi_thresh=coeff_list[0:3]
                                #best_dec_thresh=coeff_list[3:6]
                                #best_redundancy=coeff_list[6]

                                #BSB-1 EuroStress Fin
                                best_roi_thresh=[1.60712387,1.608452951,1.235778841] 
                                best_dec_thresh=[0.5,0.5,0.5]
                                best_redundancy=0.5
                                #BSB-1 EuroStress Fin
                                best_roi_thresh=[1.494683679,2.186868687,1.351940457] 
                                best_dec_thresh=[0.5,0.5,0.5]
                                best_redundancy=0.5

                                #BSB-1 180126 BSB Stress Test 2-5-10-17
                                best_roi_thresh=[1.684210526,1.780170122,1.388623073] 
                                best_dec_thresh=[0.5,0.5,0.5]
                                best_redundancy=0.5
                                #BSB-2 180126 BSB-Stress Test 2-5-10-17
                                best_roi_thresh=[1.464380649,2.207070707,1.550239234] 
                                best_dec_thresh=[0.5,0.5,0.5]
                                best_redundancy=0.5

                                #CG Model with PB Data
                                best_roi_thresh=[0.342636895268474,0.567517278043593,0.557416267942583] 
                                best_dec_thresh=[0.75,0.75,0.65]
                                best_redundancy=0.5
                           

                                #result=fun.cg_redundancy_tester(df_fin,best_roi_thresh,best_dec_thresh,best_redundancy,print_failures=False)
                                result=fun.cg_redundancy_tester_detail(df_fin,best_roi_thresh,best_dec_thresh,best_redundancy,scan_size=ROI_max,print_failures=False)
                                print filename,
                                print ";",
                                print formulation_list[0][0],
                                print ";",
                                print str(recipe[0])+"-"+str(recipe[1])+"-"+str(recipe[2]),
                                print ";",
                                print mod,
                                print ";",
                                print result[0],
                                print ";",
                                print result[1],
                                print ";",
                                print result[2],
                                print ";",
                                print result[3],
                                print ";",
                                print result[4]
                            if test_index==3:
                                df_input=copy.copy(df)
                                df_blank_input=copy.copy(df_blank)
                                formulation=formulation_list[0][0]

                                df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                                #print recipe
                                #print df_blank.iloc[:,0:5]
                                df_blank_input=fun.cg_dataframe_filter(df_blank_input,'Blank',['skip',recipe[1],'skip'],mod)
                                #print df_blank.iloc[:,0:5]
                                df_fin=fun.cg_combine_print_blank(df_input,df_blank_input)

                                #df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                                #df_fin=fun.cg_combine_print_blank(df_input,df_blank)

                           

                                result=fun.cg_white_mean_finder(df_fin)
                                print filename,
                                print ";",
                                print formulation_list[0][0],
                                print ";",
                                print str(recipe[0])+"-"+str(recipe[1])+"-"+str(recipe[2]),
                                print ";",
                                print mod,
                                print ";",
                                print result[0][0],
                                print ";",
                                print result[0][1],
                                print ";",
                                print result[0][2],
                                print ";",
                                print result[0][3],
                                print ";",
                                print result[1][0],
                                print ";",
                                print result[1][1],
                                print ";",
                                print result[1][2],
                                print ";",
                                print result[1][3]
                            if test_index==10 or test_index==11 or test_index==12 or test_index==13 or test_index==14 or test_index==15:
                                #print "t10"
                                df_input=copy.copy(df)
                                df_blank_input=copy.copy(df_blank)
                                #df_blank_input=df_blank_input.sample(n=100)
                                formulation=formulation_list[0][0]
                                if line_by_line_check:
                                    print df_input.head()
                                
                                    print path
                                    print formulation
                                    print recipe
                                    print mod

                                df_input=fun.path_filter(df_input,path)

                                df_input=fun.pk_dataframe_filter(df_input,formulation,recipe,mod)
                                size_filter='skip'
                                df_input=fun.arbitrary_filter(df_input,'size',size_filter)
                                df_input=fun.arbitrary_exclude(df_input,'formulation','blank')
                                #df_input=fun.arbitrary_exclude(df_input,'PW_check','Fail')
                                #df_input=fun.arbitrary_exclude(df_input,'MatchCheck','Fail')


                                #df_blank_input=fun.arbitrary_exclude(df_blank_input,'PW_check','Fail')
                                #df_blank_input=fun.arbitrary_exclude(df_blank_input,'MatchCheck','Fail')

                                if test_index==11 or test_index==13:
                                    #df_blank_input=fun.arbitrary_filter(df_blank_input,'Shoe',recipe[0])
                                    df_blank_input=fun.path_filter(df_blank_input,path)
                                   #df_blank_input=fun.cg_dataframe_filter(df_blank_input,'Blank',['skip',recipe[1],recipe[2]],mod)
                            
                                if line_by_line_check:
                                    print df_input.head()
                                    print df_blank_input.head()

                                #df_blank_input=fun.arbitrary_filter(df_blank_input,'Location','RIH')

                                df_fin=[]
                                df_test=[]
                                if test_index==15:
                                    train_perc=0.7
                                    msk_print = np.random.rand(len(df_input)) < train_perc
                                    msk_blank = np.random.rand(len(df_blank_input)) < train_perc
                                    df_input_train=df_input[msk_print]
                                    df_input_test=df_input[~msk_print]
                                    df_blank_input_train=df_blank_input[msk_blank]
                                    df_blank_input_test=df_blank_input[~msk_blank]
                                    df_fin=fun.cg_combine_print_blank(df_input_train,df_blank_input_train) 
                                    df_test=fun.cg_combine_print_blank(df_input_test,df_blank_input_test)
                                else:
                                    df_fin=fun.cg_combine_print_blank(df_input,df_blank_input) 
                     
                                #df_fin=fun.arbitrary_exclude(df_fin,'PW_check','Fail')
                                #df_fin=fun.arbitrary_exclude(df_fin,'MatchCheck','Fail')
                               # df_fin=fun.arbitrary_exclude(df_fin,'NikeCheck','Fail')
                                #df_fin=fun.arbitrary_filter(df_fin,'Path','180303')

                                df_list=[]
                                for pFilter in ['180304','180303','180302']:
                                    df_list.append(fun.arbitrary_filter(df_fin,'path',pFilter))
                                df_fin = pd.concat(df_list)


                                if line_by_line_check:
                                    print df_fin
                                    print len(df_fin)
                               # df_fin=fun.arbitrary_filter(df_fin,'Flash','NoFlash')
                                df_fin=fun.dropNAN(df_fin,10)
                                
                             #   print len(df_fin)
                                if line_by_line_check:
                                    print df_fin

                                if test_index==10 or test_index==11 or test_index==14 or test_index==15:
                                    result=fun.pk_modeler(df_fin,ring,[2]);

                                   #add additional test data to result
                                    result=result.assign(AP=pd.Series([mod]*result.shape[0]).values) #mod (AP)
                                    result=result.assign(Shoe=pd.Series([recipe[0]]*result.shape[0]).values) #Shoe
                                    result=result.assign(Brand=pd.Series([recipe[1]]*result.shape[0]).values) #Brand
                                    result=result.assign(Locaiton=pd.Series([recipe[2]]*result.shape[0]).values) #Location
                                    result=result.assign(Ring_count=pd.Series([ring]*result.shape[0]).values) #Ring_count
                                    result=result.assign(Formulation=pd.Series([formulation]*result.shape[0]).values) #Formulation
                                    result=result.assign(Path=pd.Series([path]*result.shape[0]).values) #Path filter
                                    result=result.assign(Size=pd.Series([size_filter]*result.shape[0]).values) #Size filter
                                    result=result.reset_index(drop=True)

                                    result.to_csv("DF Export -"+filename+"- Form_"+formulation+"_"+recipe[0]+"_"+recipe[1]+"_"+recipe[2]+" AP_"+mod+"__"+str(ring)+" Path_"+path+".csv")

                                    #print result['J_Abs']
                                    #print result
                                    result_max=result.ix[result['J_Abs'].idxmax()]

                            
                                    print list(result_max)

                                    if test_index==14 or test_index==15:
                                        #make a dict with the top n values
                                        max_number=20
                                        df_big=result.sort_values(['J'], ascending=False)[:max_number]
                                        df_new=df_big[['Test_Name','Thresh']]
                                        #print df_big
                                        df_new=df_new.reset_index(drop=True)
                                        df_new
                                        thresh_dict={}
                                        for iter in range(max_number):
                                            thresh_dict[df_new.iloc[iter]["Test_Name"]]=float(df_new.iloc[iter]["Thresh"])
                                      #  print thresh_dict
                                        max_J,best_sen,best_spec,count_blank,count_print,best_red,best_guess_mat,fail_pic_list=fun.pk_redundancy_tester(df_fin,thresh_dict,[1])
                                        print max_J,
                                        print ";",
                                        print best_sen,
                                        print ";",
                                        print best_spec,
                                        print ";",
                                        print count_blank,
                                        print ";",
                                        print count_print,
                                        print ";",
                                        print best_red,
                                        print ";",
                                        print path,
                                        print ";",
                                        print formulation,
                                        print ";",
                                        print recipe,
                                        print ";",
                                        print mod,
                                        print ";",
                                        print best_guess_mat

                                        if test_index==15:
                                            max_J,best_sen,best_spec,count_blank,count_print,best_red,best_guess_mat,fail_pic_list=fun.pk_redundancy_checker(df_test,best_guess_mat,best_red,[1])
                                            print max_J,
                                            print ";",
                                            print best_sen,
                                            print ";",
                                            print best_spec,
                                            print ";",
                                            print count_blank,
                                            print ";",
                                            print count_print,
                                            print ";",
                                            print best_red,
                                            print ";",
                                            print path,
                                            print ";",
                                            print formulation,
                                            print ";",
                                            print recipe,
                                            print ";",
                                            print mod,
                                            print ";",
                                            print best_guess_mat

                                elif test_index==12 or test_index==13:
                                    #0.75in 120mS
                                    thresh_dict={"R3_minus_R0":-0.302181647059,
                                                 "R4_minus_R0":-0.134583089286,
                                                 "R5_minus_R0":0.207344540541,
                                                 "R6_minus_R0":0.620065144578,
                                                 "R7_minus_R0":0.22301403125,
                                                 "R8_minus_R0":0.311315090909,
                                                 "R9_minus_R0":0.587751808219,
                                                 "R10_minus_R0":0.680360333333,
                                                 "R11_minus_R0":0.83656889899
                                                 }

                                    #0.75in 80mS
                                    thresh_dict={"R3_minus_R0":0.209758272727,
                                                 "R4_minus_R0":-0.191298140625,
                                                 "R5_minus_R0":0.205945771429,
                                                 "R6_minus_R0":0.3161038,
                                                 "R7_minus_R0":0.341899721519,
                                                 "R8_minus_R0":0.19279075,
                                                 "R9_minus_R0":0.200890606742,
                                                 "R10_minus_R0":0.573791195122,
                                                 "R11_minus_R0":-0.0323681818182
                                                 }

                                    # #0.5in 120mS
                                    #thresh_dict={"R3_minus_R0":-0.553791485714,
                                    #             "R4_minus_R0":-0.4557828,
                                    #             "R5_minus_R0":-0.258344205128,
                                    #             "R6_minus_R0":-0.20403734,
                                    #             "R7_minus_R0":-0.00533163157895,
                                    #             "R8_minus_R0":0.0963742592593,
                                    #             "R9_minus_R0":0.580236829268,
                                    #             "R10_minus_R0":0.293244539474,
                                    #             "R11_minus_R0":0.422045171717
                                    #             }

                                    #0.75in 80mS 13'17'512
                                    thresh_dict={'R6_minus_R4':0.3857748,'R6_minus_R3':0.461645181818,'R7_minus_R1':0.181067704225,'R7_minus_R2':0.434563809524,'R7_minus_R3':0.262770043478,'R6_minus_R1':0.214561125,'R6_minus_R2':0.279850692308,'R8_minus_R3':0.566323608696,'R7_minus_R4':0.259943487805,'R9_minus_R3':0.559671869565}

                                    #0.75in 80mS 13'17'512 - More Data
                                    thresh_dict={'R6_minus_R2':0.324311350649,'R7_minus_R3':0.2823504,'R6_minus_R3':0.272505131579,'R7_minus_R2':0.531834425287,'R6_minus_R1':0.297184756757,'R5_minus_R3':0.296657973684,'R7_minus_R1':0.378510170732,'R6_minus_R0':-0.0300233030303,'R7_minus_R0':0.193383789474,'R6_minus_R4':0.29942506383}

                                     #0.75in 120mS Sponge 18'13'17'23'512 All Shoes, Jordan Blank, EV0
                                    thresh_dict={'R3_minus_R1':0.264797142857,'R4_minus_R1':0.505784684211,'R3_minus_R1':0.264797142857,'R4_minus_R1':0.505784684211,'R4_minus_R2':0.38302,'R5_minus_R1':0.209633043478,'R6_minus_R1':0.286477047619,'R3_minus_R2':0.298588,'R6_minus_R1':0.286477047619,'R3_minus_R2':0.298588}

                                    #PK_180301_HighConcSweep_13'18'17'23'512.csv- Form_200mS_0.5 inch_skip_skip AP_skip__12 Path_skip
                                    thresh_dict={'R9_minus_R1':2.40817,'R9_minus_R0':3.16156481818,'R8_minus_R0':2.23796124242,'R10_minus_R1':2.52427334343,'R10_minus_R0':2.77649522222,'R8_minus_R1':1.72043013978,'R7_minus_R0':1.84395175758,'R11_minus_R0':2.80667457576,'R6_minus_R0':1.83736493939,'R7_minus_R1':1.5320740404,'R11_minus_R1':2.18822708081,'R6_minus_R1':1.80962318182}
                                    
                                    #PK_180301_HighConcSweep_13'18'17'23'512.csv- Form_150mS_0.5 inch_skip_skip AP_skip__12 Path_skip
                                    thresh_dict={'R8_minus_R0':2.23728275294,'R9_minus_R0':2.24474215152,'R7_minus_R0':1.78910909677,'R7_minus_R1':1.71260636364,'R10_minus_R0':2.29457262626,'R8_minus_R1':2.11878215789,'R9_minus_R1':2.17563355556,'R10_minus_R1':2.2431310404,'R6_minus_R0':1.64037050538,'R6_minus_R1':1.71993181633,'R11_minus_R0':2.21012445455,'R5_minus_R1':1.30557436842}

                                    #PK_180301_HighConcSweep_18'17'23'512.csv- Form_200mS_0.5 inch_skip_skip AP_skip__12 Path_skip
                                    thresh_dict={'R9_minus_R1':2.80147738384,'R8_minus_R0':2.46280125253,'R9_minus_R0':3.06900615152,'R10_minus_R1':2.62973077778,'R8_minus_R1':1.95829775758,'R7_minus_R1':1.6649879798,'R10_minus_R0':3.20375165657,'R7_minus_R0':1.84404831313,'R6_minus_R0':1.9633759798,'R11_minus_R0':2.79371392929,'R9_minus_R2':2.52307120202,'R11_minus_R1':2.11950738384}

                                    #PK_180301_HighConcSweep_18'17'23'512.csv- Form_150mS_0.5 inch_skip_skip AP_skip__12 Path_skip
                                    thresh_dict={'R8_minus_R0':2.39430371717,'R7_minus_R0':1.90705291919,'R9_minus_R1':2.03929869697,'R9_minus_R0':2.27220827273,'R8_minus_R1':2.15828050505,'R7_minus_R1':1.99712336364,'R10_minus_R1':2.25375409091,'R10_minus_R0':2.55636686869,'R6_minus_R1':1.37059941935,'R6_minus_R0':2.12828712195,'R11_minus_R1':2.33423427273,'R11_minus_R0':2.16216768687}

                                    #PK_180302_GlareMask_18'17'23'650'512._AND_PK_180302_DemoSet2_18'17'23'650'512.csv Adj J
                                    thresh_dict={'R7_minus_R1':2.34001503,'R9_minus_R0':2.652195798,'R8_minus_R0':2.514607889}
                                    red_val=1.5

                                    #PK_180302_SUPEREEE_27'17'23'651'512.csv
                                    thresh_dict= {'R10_minus_R1': 2.3355341414141435,'R10_minus_R0':4.54737984848480,'R9_minus_R1': 3.39460045454544,'R8_minus_R1': 2.7685268181818143,'R8_minus_R0': 3.0356904141414125}
                                    red_val=1.5

                                    max_J,best_sen,best_spec,count_blank,count_print,best_red,best_guess_mat,fail_pic_list=fun.pk_redundancy_checker(df_fin,thresh_dict,red_val,[1])
                                           
                                   # max_J,best_sen,best_spec,count_blank,count_print,best_red,best_guess_mat,fail_pic_list=fun.pk_redundancy_tester(df_fin,thresh_dict,[1])
                                    print max_J,
                                    print ";",
                                    print best_sen,
                                    print ";",
                                    print best_spec,
                                    print ";",
                                    print count_blank,
                                    print ";",
                                    print count_print,
                                    print ";",
                                    print best_red,
                                    print ";",
                                    print path,
                                    print ";",
                                    print formulation,
                                    print ";",v
                                    print recipe,
                                    print ";",
                                    print mod,
                                    print ";",
                                    print best_guess_mat,
                                    #print ";",
                                    #print fail_pic_list

                        except:
                            skipped=True
if(1): #Genertic Thresholding Analysis
    
    #0 - Black and White Simple Check
    #1 - Black and White Redundancy Analysis
    #2 - Circle Check Redundancy Analysis
    #3 - Draw Histogram
    #4 - H Combine ROIs in Histogram
    #5 - H Turn OFF Divided ROI Histogram
    #6 - H Save figure rather than show
    #7 - H Draw or Show Histogram Filter by Filter
    #8 - Add Binning to Circle Check
    #9 - Serial Test for Circle Check
    #10 - Use ALL ROIs for Circle Check


    #20 - Rings Simple Check

    test_index=[-1,2,3,6,7]
    test_index=[2,9]

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
    active_serial=2


    ROI_scans=10
    ROI_count=6

    line_by_line_check=0


    files=[]
    start_dir='csv_import'
    pattern   = "*.csv"
    print 'prog1'
    for dir,_,_ in os.walk(start_dir): 
        files.extend(glob(os.path.join(dir,pattern))) 
    for file in files:
        if 3 in test_index:
            label_list=[]
        #print 'prog2'
        filename=file.rpartition("\\")[2]
        #Import csv to dataframe only once

        if (line_by_line_check):
            print filename

        #grab ring number from filename
        f_list=filename.split("_")
        end_bit=f_list[-1]
        ring_split=end_bit.split("'")
        ring_string=ring_split[-1]
        ring_string=ring_string[:-4]
        #ring=55
        try:
            ring=int(ring_string)-500
        except:
            ring=55
        #ring=36

        number_filters=np.logspace(2,5,50)
        number_filter=0
        number_filters=[400,700,900,1000,1500,2500]
        #number_filters=range(3)
        number_filter=0

        for active_serial in [1,2]:
        #for number_filter in number_filters:
        #if True:

            df=pd.read_csv(file,header=0,error_bad_lines=False,warn_bad_lines=False)

            if line_by_line_check:
                print "ORIGINAL DATAFRAME: "
                print df


            #drop error rows
            #df=df.loc[df['guess'].isin([0,1])]
            df=df.loc[df['guess'].isin([-99])]

            df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
            if line_by_line_check:
                print "DROP UNNAMED: "
                print df



            #drop empty colulmns
            df=df.dropna(axis=1,how='all')



            #df=fun.dropNAN(df,41)
            if line_by_line_check:
                print "DROP NAN VALUES: "
                print df
            #print df

            #Manual number filters
            #df=fun.arbitrary_include_number(df,'exposurebias',-1.4,0.2)
            #df=fun.arbitrary_include_number(df,'lux',number_filter,5)
            #df=fun.arbitrary_include_number(df,'lux',number_filter,0.2,keep_all_low=True,keep_all_high=False)
            #df=fun.arbitrary_include_number(df,'lux',number_filter,1000)

            #Manual text filter
            #df=fun.arbitrary_exclude(df,'location','window')
            path_list=['180504 Perry_15Short_iP8plus','180504 Perry_15Short_iP7','180504 Ben_15Short_iP8_HeadOn','skip']
            #df=fun.arbitrary_include(df,'path',path_list[number_filter])
            #df=fun.arbitrary_include(df,'path','Perry_15Short_iP8plus')
            #df=fun.arbitrary_exclude(df,'lux','900')
            #df=fun.arbitrary_exclude(df,'lux','2500')
            #df=fun.arbitrary_exclude(df,'lux','1500')

            #If serial check - modify columns
            if 9 in test_index:
                #Reset data index
                df=df.reset_index(drop=True)
                #First make a DF with no ROI columns
                changed_df=copy.copy(df)
                cols_no_roi=[c for c in changed_df.columns if c.lower()[:3]!='roi']

                changed_df=changed_df[cols_no_roi]
                #print changed_df
                ##Then add back ROI columns if their name has been changed
                #make 
                active_rois=[]
                for key in serial_dict.keys():
                    if serial_dict[key][0]==active_serial:
                        active_rois.append([key,serial_dict[key][1]])
                for rename_pair in active_rois:
                    for col_name in df.columns:
                        if 'roi_'+str(int(rename_pair[0])) in col_name:
                            new_col_name=col_name.replace('roi_'+str(int(rename_pair[0])),'roi_'+str(int(rename_pair[1])))
                            changed_df[new_col_name]=df[col_name]
                df=copy.copy(changed_df)
 
                #print changed_df

            #Make Dataframe of Blanks
            df_blank=df.loc[df['mark']==0]
            cap_list=['blank','blankout','blankfill']
            #df_blank=fun.arbitrary_filter(df,'cap',cap_list[number_filter])

            if line_by_line_check:
                print "BLANK DATAFRAME"
                print df_blank

            #filter blank dataframe
            #df_blank=fun.arbitrary_include(df_blank,'numberofprints','2')

            #Make Dataframe of Prints
            df=df.loc[df['mark']==1]

            #df=fun.arbitrary_filter(df,'mark','1')

            #####Exclude List#####
            exclude_list=[
                ]


             #####Exclude List Combo#####
            exclude_list_combo=[
                ]

            #Unique include
            unique_key='test'
            unique_list=df[unique_key].unique()

            ####Include List####
            include_list=[
                ['path',['skip']]
                #['serial',['1','2']]
               #['path',['180505 Cap15_iP8plus_2photo','180505 Cap15_iP8_2Photo','180505 Cap15_iP7','skip']],
           #    ['path',['iP8','iP7']],
          #     [unique_key,unique_list]
               #['location',['whitetable','blacktable','window']],
            #   ['location',['window']],
            #   ['lux',['100']]
             #  ['test',['stcboardroomphotoset']]
            #     ['cap',['cap4']],
            #    ['scenario',['campus']],
            #    ['test',['cap2photoset']]
            #    ['background',['black','white','window','skip']],
            #    ['lux',['555','1400','1650','1700','3000','3200','6200','8000','11000','11700','34000']]
          #  ['lux',['440','770','1400','2600','3400']],
            #['expbias',['-2.4','-1.9','-1.4']]
        
                ]

            ####Include List Combo####
            include_list_combo=[
                ]

            #Transform Include and Exclude lists into product lists
            include_array=[]
            include_array_head=[]
            for include_pair in include_list:
                include_array.append(include_pair[1])
                include_array_head.append(include_pair[0])

            include_product_list=list(it.product(*include_array))

            exclude_array=[]
            exclude_array_head=[]
            for exclude_pair in exclude_list:
                exclude_array.append(exclude_pair[1])
                exclude_array_head.append(exclude_pair[0])

            exclude_product_list=list(it.product(*exclude_array))

            if line_by_line_check:
                print include_product_list
                print exclude_product_list

            ROI_list=range(3)
            #x_col_list=[] #these should be COLUMN HEADERS
            #x_col_list.append()

            if True:
            #try:
                if len(exclude_list_combo)>0:
                    for exclude_pair in exclude_list_combo:
                        for exclude_string in exclude_pair[1]:
                            if line_by_line_check:
                                print exclude_string,
                                print " ",
                                print exclude_pair[0]
                            df=fun.arbitrary_exclude(df,exclude_pair[0],exclude_string)
                if line_by_line_check:
                    print df
                if len(include_list_combo)>0:
                    for include_pair in include_list_combo:
                        df_concat=[];
                        for include_string in include_pair[1]:
                            if line_by_line_check:
                                print include_string,
                                print " ",
                                print include_pair[0]
                            df_concat.append(fun.arbitrary_include(df,include_pair[0],include_string))
                        df=pd.concat(df_concat)
                df_pre_filter=copy.copy(df)
                df_blank_pre_filter=copy.copy(df_blank)

                if len(include_list)>0:
                
                    for include_line in include_product_list:
                        include_line_print=""
                        if line_by_line_check:
                            print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                            print include_line
                        #try:
                        if True:
                            parameter_export=""
                            df=copy.copy(df_pre_filter)
                            df_blank=copy.copy(df_blank_pre_filter)
                        
                            for include_index in range(len(include_line)):
                                df=fun.arbitrary_include(df,include_array_head[include_index],include_line[include_index])
                                if -1 in test_index:
                                    df_blank=fun.arbitrary_include(df_blank,include_array_head[include_index],include_line[include_index])
                                parameter_export=parameter_export+include_array_head[include_index]+"'"+include_line[include_index]+"_"
                                include_line_print=include_line_print+include_array_head[include_index]+":"+include_line[include_index]+";"
                                if line_by_line_check:
                                    print "______________________________________________________________________"
                                    print include_array_head[include_index]+";"+include_line[include_index]
                                    print df
                                    print "______________________________________________________________________"          
                            if len(exclude_list)>0:
                                for exclude_line in exclude_product_list:
                                    for exclude_index in range(len(exclude_line)):
                                        df=fun.arbitrary_exclude(df,exclude_array_head[exclude_index],exclude_line[exclude_index])

                   

                            #modify input due to serial stuff
                            if 9 in test_index:
                                df_print_blank=fun.arbitrary_exclude(df,'serial',str(active_serial))
                                df_print_print=fun.arbitrary_include(df,'serial',str(active_serial))
                                df=copy.copy(df_print_print)
                                df_blank=pd.concat([df_print_blank,df_blank])
                            #prep input to thresholding functions
                            df_input=copy.copy(df)
                            df_blank_input=copy.copy(df_blank)
                            df_fin=fun.cg_combine_print_blank(df_input,df_blank)
                            if line_by_line_check:
                                print "PRINT DATAFRAME:"
                                print df_input
                                print "BLANK DATAFRAME:"
                                print df_blank
                                print "FINAL DATAFRAME:"
                                print df_fin
                                print df_fin.mark
                            #make include_line that can be used as a filename
                            include_line_filename=string.replace(include_line_print,':',' ')
                            include_line_filename=string.replace(include_line_filename,';',',')

                            if 0 in test_index: #black/white analysis
                                if line_by_line_check:
                                    print "black and white analysis"
                                test_list=[2]
                                white_count=1
                                for white_index in range(white_count):
                                    result,max_result=fun.black_white_modeler(df_fin,white_index,1,test_list)

                                    df_one_line=pd.DataFrame.from_records([include_line],columns=include_array_head)
                                    df_multiline=copy.copy(df_one_line)

                                    for i in range(result.shape[0]-1):
                                        df_multiline=pd.concat([df_one_line,df_multiline])

                                   # df_descriptor=df_descriptor.append(df_descriptor*result.shape[0],ignore_index=True)

                                    df_multiline=df_multiline.reset_index(drop=True)
                                    result=result.reset_index(drop=True)

                                    result_fin=pd.concat([df_multiline,result],axis=1)
                                  #  result_fin=result_fin.reset_index(drop=True)

                                    if line_by_line_check:
                                        print result_fin

                                    #for head_index in range(len(include_array_head)):
                                     #   result=result.assign(include_array_head[head_index]=pd.Series([include_line[head_index]]*result.shape[0]).values) #Formulation

                                    result_fin.to_csv("DF-"+filename+" "+parameter_export+str(test_list)+".csv")


                                    result_max=result_fin.ix[result_fin['j_abs'].idxmax()]   
                                    result_max_list=list(result_max)
                                    result_max_list=[filename]+result_max_list
                                    print result_max_list
                            if 1 in test_index:
                                if line_by_line_check:
                                    print "black and white redundancy analysis"
                                test_list=[4]
                                white_count=3
                                black_count=5
                                result=fun.black_white_redundancy(df_fin,white_count,black_count,test_list)
                                #print "finished"
                                export_cols=["j","sen","spec","active_rois","red_thresh","n_b","n_p"]
                                roi_opt_export_cols=["white_index","white_bin","black_index","black_bin","thresh","j","test_name"]
                                for thresh_index in range(white_count):
                                    export_cols.append("roi"+str(int(thresh_index)))
                                #print result
                            
                                print filename,
                                print ";",
                                for col in export_cols:
                                    print result[col],
                                    print ";",
                                #print result["optimal_rois"].

                                for col in roi_opt_export_cols:
                                    print col+":",
                                    for ROI_index in result["active_rois"]:
                                
                                        print str(result["optimal_rois"][ROI_index][col])+",",
                                    print ";",
                                print include_line_print

                            if 2 in test_index:
                                if line_by_line_check:
                                    print "circle check redundancy analysis"

                                if 8 in test_index:
                                    bin_inputs=range(55)
                                else:
                                    bin_inputs=[16]

                               
                                for bin_index in bin_inputs:
                                #if True:
                                    if 8 in test_index:
                                        string_input="_bin_"+str(int(bin_index))
                                    else:
                                        string_input=""
                                    if 10 in test_index:
                                        use_all_ROIs=True
                                    else:
                                        use_all_ROIs=False

                                    result=fun.cg_redundancy_modeler_v4(df_fin,ROI_scans,ROI_count,string_input,use_all_ROIs)
                                    #result=fun.cg_redundancy_modeler_v4(df_fin,ROI_scans,ROI_count)

                                    export_cols=["j","sen","spec","active_rois","red_thresh","n_b","n_p","t","p"]
                                    roi_opt_export_cols=["white_index","thresh","dec_thresh","j"]
                                    for thresh_index in range(ROI_count):
                                        export_cols.append("roi"+str(int(thresh_index)))
                                    #print result
                            
                                    print filename,
                                    print ";",
                                    for col in export_cols:
                                        print result[col],
                                        print ";",
                                    #print result["optimal_rois"].

                                    for col in roi_opt_export_cols:
                                        print col+":",
                                        for ROI_index in result["active_rois"]:
                                
                                            print str(result["optimal_rois"][ROI_index][col])+",",
                                        print ";",
                                    print "bin:"+str(int(bin_index))+";",
                                    print "numfilter:"+str(number_filter)+";",
                                   # print str(number_filter)+";",
                                    if 9 in test_index:
                                        print "active_serial:"+str(active_serial)+";",
                                    print include_line_print

                            if 3 in test_index:
                                if line_by_line_check:
                                    print "data histogram"
                                #print df_fin
                                analysis_dict={}
                                analysis_dict["active_rois"]=range(ROI_count)
                                analysis_dict["roi_scans"]=ROI_scans
                                analysis_dict["include_line_print"]=include_line_print

                                histo_data=fun.data_combo(df_fin,analysis_dict,"circle")

                                #print histo_data["blank_roi0"]

                                roi_combo_list=[]
                                histo_combo_dict={}
                                if 4 in test_index:
                                    for key in histo_data:
                                        split_key=key.split("_roi")
                                        if split_key[0] in roi_combo_list:
                                            histo_combo_dict[split_key[0]].extend(histo_data[key])
                                        
                                        else:
                                            histo_combo_dict[split_key[0]]=histo_data[key]
                                            roi_combo_list.append(split_key[0])
                                    for key in histo_combo_dict:
                                        if not key in label_list:
                                            data=histo_combo_dict[key]
                                            if len(data)>0:
                                                plt.hist(histo_combo_dict[key],100,alpha=0.3,label=key,normed=1,histtype='stepfilled')
                                                label_list.append(key)
                                if not 5 in test_index:
                                    for key in histo_data:
                                        if not key in label_list:
                                            data=histo_data[key]
                                            if len(data)>0:
                                                plt.hist(histo_data[key],20,alpha=0.3,label=key,normed=1,histtype='stepfilled')
                                                label_list.append(key)

                                if 7 in test_index:
                                    print include_line_filename
                                    label_list=[]
                                    plt.legend()
                                    plt.xlim(-10,35)
                                    plt.title(filename)
                                    if 6 in test_index:
                                        plt.savefig('H_'+filename[:-4]+include_line_filename+'.jpg', format='jpg', dpi=400)
                                    else:
                                        plt.show()
                                    plt.clf()
                                        #print key
                                        #print len(histo_data[key])
                                        #print histo_data[key]
                                    #plt.legend()
                                    #plt.show()
                                    #print data
                                    #for iter in range(len(data)):
                                    #    print len(data[iter])
                                    #plt.hist(data,100,alpha=0.3,normed=0,histtype='stepfilled')
                                    #plt.legend(keys)
                                    #plt.show()
                            if 20 in test_index: #black/white analysis
                                if line_by_line_check:
                                    print "ring analysis"
                                test_list=[6]
                                white_count=1
                                for roi_index in range(white_count):
                                    result,max_result=fun.rings_modeler(df_fin,ring,test_list,roi_index)

                                    df_one_line=pd.DataFrame.from_records([include_line],columns=include_array_head)
                                    df_multiline=copy.copy(df_one_line)

                                    for i in range(result.shape[0]-1):
                                        df_multiline=pd.concat([df_one_line,df_multiline])

                                   # df_descriptor=df_descriptor.append(df_descriptor*result.shape[0],ignore_index=True)

                                    df_multiline=df_multiline.reset_index(drop=True)
                                    result=result.reset_index(drop=True)

                                    result_fin=pd.concat([df_multiline,result],axis=1)
                                  #  result_fin=result_fin.reset_index(drop=True)

                                    if line_by_line_check:
                                        print result_fin

                                    #for head_index in range(len(include_array_head)):
                                     #   result=result.assign(include_array_head[head_index]=pd.Series([include_line[head_index]]*result.shape[0]).values) #Formulation

                                    result_fin.to_csv("DF-"+filename+" "+parameter_export+str(test_list)+".csv")


                                    result_max=result_fin.ix[result_fin['j_abs'].idxmax()]   
                                    result_dict=result_max.to_dict()
                                    result_max_list=list(result_max)
                                    result_max_list=[filename]+result_max_list

                                    export_cols=["j","sen","spec","white_index","white_bin","black_index","black_bin","n_b","n_p"]
                                    print filename,
                                    print ";",
                                    for col in export_cols:
                                        print result_dict[col],
                                        print ";",
                                    print include_line_print
                    #    except:
                     #       skipped=True
            if 3 in test_index:
                if not 7 in test_index:
                    plt.legend()
                    plt.title(filename+"_"+str(number_filter))
                    #plt.title(filename)
                    plt.xlim(-5,5)
                    if 6 in test_index:
                        plt.savefig('H_'+filename[:-4]+'NF_'+str(int(number_filter))+'.jpg', format='jpg', dpi=400)
                    else:
                        plt.show()
                    plt.clf()
                    label_list=[]
                        
                            



if(0): #FeatureFinding Accuracy Test
    test_index=[]

    point_count=6

    line_by_line_check=0


    files=[]
    json_files=[]
    start_dir='csv_import'
    start_json_dir='json_import'
    pattern   = "*.csv"
    json_pattern="*.json"
    print 'prog1'
    for dir,_,_ in os.walk(start_dir): 
        files.extend(glob(os.path.join(dir,pattern))) 
    for dir,_,_ in os.walk(start_json_dir): 
        json_files.extend(glob(os.path.join(dir,json_pattern))) 
    
    std_dict={}
    for json_file in json_files:
        with open(json_file) as json_data:
            adder_dict=json.load(json_data)
        std_dict.update(adder_dict)

    #Delete keys that don't the the correct number of points
    for key in std_dict.keys():
        if not len(std_dict[key])==point_count:
            del std_dict[key]

    for file in files:


        filename=file.rpartition("\\")[2]
        #Import csv to dataframe only once

        if (line_by_line_check):
            print filename



        df=pd.read_csv(file,header=0,error_bad_lines=False,warn_bad_lines=False)

        if line_by_line_check:
            print "ORIGINAL DATAFRAME: "
            print df

        df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if line_by_line_check:
            print "DROP UNNAMED: "
            print df

        df=fun.arbitrary_include(df,'path','iP8plus')

        distance_superdict={}
        print filename
        print '{} {} {} {}'.format('point','avg','stdev','count')
        for ff_index in range(point_count):
            mean_dist,stdev_dist,list_dist,dict_dist=fun.ff_accuracy(df,std_dict,ff_index)
            print '{0:2d} {1:3f} {2:4f} {3:5d}'.format(ff_index,mean_dist,stdev_dist,len(list_dist))
            #print stdev_dist
            distance_superdict[str(ff_index)]=list_dist
        #    for key in dict_dist.keys():
        #        print key

        #print distance_superdict
        #for key in distance_superdict[0].keys():
        #    print key

        for key in distance_superdict:
            data=distance_superdict[key]
            if len(data)>0:
                plt.hist(distance_superdict[key],alpha=0.3,label=key,normed=1,histtype='stepfilled')
                #label_list.append(key)
       
                        
                            

if(0): #csv import and sort
    csv_file="G://Developer//csvExports//InitialCP Sweep//Combo_data.csv"
    csv_file="G://Developer//csvExports//171124 Distance Test//Combo_data.csv"
    csv_file="G://Developer//csvExports//171124 TriPic Test//Combo_data.csv"
    fun.import_and_sort_csv(csv_file,1000,0)

if(0): #make a list
    numbers=['brightness','luxpred','luspredcorr','avg','blue','lux']
    start_number_of_digits=1
    print "{",
    for iter in range(start_number_of_digits,len(numbers)+1):
    #for iter in range(start_number_of_digits,4):
        for val in it.combinations(numbers,iter):
            x_cols=list(val)
           # x_cols.extend([15,35])
            print "{",
            for iter in range(len(x_cols)-1):
                print x_cols[iter],
                print ";",
            print x_cols[-1],
            print "},",
    print "}"

  
if(0): #make a text list for python
    numbers=['blue_avg','red_avg','green_avg','value_avg','grey_avg']
    start_number_of_digits=1
    print "[",
    for iter in range(start_number_of_digits,len(numbers)+1):
    #for iter in range(start_number_of_digits,4):
        for val in it.combinations(numbers,iter):
            x_cols=list(val)
            x_cols.extend(['brightness'])
            print "[",
            for iter in range(len(x_cols)-1):
                print "'",
                print x_cols[iter],
                print "',",
            print "'",
            print x_cols[-1],
            print "'],",
    print "]"

if(0): #make a simple list
    numbers=[503,505,507,509,511,521,519]
    start_number_of_digits=1
    print "{",
    #for iter in range(start_number_of_digits,len(numbers)+1):
    for number in numbers:
        x_cols=[number]
        x_cols_temp=[2031,17,35]
        x_cols_temp.extend(x_cols)
        x_cols=copy.copy(x_cols_temp)
       # x_cols.extend()
        print "{",
        for iter in range(len(x_cols)-1):
            print x_cols[iter],
            print ",",
        print x_cols[-1],
        print "},",
    print "}"
    #print result
    #for val in result[3]:
    #    for val2 in val:
    #        print str(val2)+";",