#Run Functions
print "importing..."
import cv2
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
    start_dir="G://Google Drive//Original Images//Pumped Kicks//180302 Sponge_Stamp_Diff_Lux"

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
if(0): #Logistic Regression
    tn=1;
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
                print str(perc)+","+str(result[0])+","+str(result[1])+","+str(result[2])
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
                #    print ",",
                #print " "
                    #print row_keep
                    #print str(perc)+","+str(result[0])+","+str(result[1])+","+str(result[2])
                    #print perc,
                    #print ",",
                    #print result[4][0],
                    #print ",",
                    #for val in result[3]:
                    #    for val2 in val:
                    #        print str(val2)+",",
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
                #print str(result[0])+","+str(result[1])+","+str(result[2])
                #print result[4][0],
                #print ",",
                #for val in result[3]:
                #    for val2 in val:
                #        print str(val2)+",",
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
                    print ",",
                    print x_cols,
                    print ",",
                    print result
                    #print "anal_done"
                    #result.extend(x_cols)
                    #with open(r'python_export_data_'+str(iter)+'superData.csv', 'ab') as f:
                    #    writer = csv.writer(f)
                    #    writer.writerow(result)
                except:
                    print "Failed x_col: ",
                    print x_cols
                #print str(result[0])+","+str(result[1])+","+str(result[2])
                #print result[4][0],
                #print ",",
                #for val in result[3]:
                #    for val2 in val:
                #        print str(val2)+",",
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
                    print ",",
                    print x_cols,
                    print ";",
                    x_train,y_train,x_test,y_test=fun.logistic_regression_prep(csv_file,x_cols,row_keep,0.66)
                    result=fun.logistic_regression_model(x_train,y_train,printer=False)
                    print ",",
                    print str(result[0])+","+str(result[1])+","+str(result[2])
                    print ","
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
        print str(result[0])+","+str(result[1])+","+str(result[2])
if(1): #CanadaGoose Logistic Regression
    #0 - logistic regression
    #1 - threshold modeling
    #2 - threshold model testing
    #3 - mean finder
    #4 - threshold modeling v2
    #5 - threshold modeling v2 with filtered Blanks and Detailed Report
    #6 - threshold modeling v3 - arbitrary ROI_max and ROI_count
    test_index=14

    line_by_line_check=0

    files=[]
    start_dir='C://Users//gttho//Documents//Visual Studio 2017//Projects//PythonApplication1//PythonApplication1//csv_import'
    pattern   = "*.csv"
    print 'prog1'
    for dir,_,_ in os.walk(start_dir): 
        files.extend(glob(os.path.join(dir,pattern))) 
    for file in files:
        #print 'prog2'
        filename=file.rpartition("\\")[2]
        #Import csv to dataframe only once

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

        df=fun.set_column_sequence(df,['path','formulation','ap','shoe','brand','location','size','colour','lux','datetime'])
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

        #print df

        #Make Dataframe of Blanks
        df_blank=df.loc[df['formulation'].str.contains('blank|Blank')]

        #df_blank=df.loc[df['Path'].str.contains('blank|Blank')]

        if line_by_line_check:
            print df_blank


        ##############_PATH FILTERS_#####################
        path_list=[]
       # path_list.append('180221 Tag_Variations_80mS_AP1')
       # path_list.append('Yeezy')

     #   path_list.append('180228')
      #  path_list.append('180301')
        path_list.append('skip')
    #    path_list.append("Solid_Square_80mS_100mS")

      #  path_list=["skip"]

        ###############_FORMULATOINS_####################
        formulation_superlist=[]
        #Add formulations in the format [FORMULATION,[[INK,BINDER,SOLVENT]]] or [[FORMULATION,[[INK,BINDER,SOLVENT],[INK2,BINDER2,SOLVENT2]]]
        #formulation_superlist.append([['P.FM1 - 50/50-C - PM'],[[3.5,66.5,30],[3,67,30]]])
        #formulation_superlist.append([['S.FM1 - 50/50-C - PM'],[[0.5,75.5,25],[1,72,27]]])

        #formulation_superlist.append([['BR Formulation'],[[50,10,10],[50,10,11],[50,10,20]]])
        #formulation_superlist.append([['P.FM1 - 50/50-B - PM'],[[10,65,25],[6.5,67,26.5]]])
        #formulation_superlist.append([['P.FM1 - 50/50-C - PM'],[[10,65,25]]])
        #formulation_superlist.append([['S.FM1 - 50/50-B - PM'],[[1.25,73,25.75],[1.5,73.5,25]]])
        #formulation_superlist.append([['S.FM1 - 50/50-C - PM'],[[0.5,70.5,29],[0.5,74.5,25],[0.5,76.5,23]]])
        
        #formulation_superlist.append([['BR Formulation'],[[50,10,10]]])

        #formulation_superlist.append([['S.FM1 - 50/50-B - PM'],[[1.25,73,25.75]]])

        #formulation_superlist.append([['Ink'],[[750,0,0]]])

        #formulation_superlist.append([['Ink'],[[750,1,0]]])
        #formulation_superlist.append([['Ink'],[[750,2,0]]])
        #formulation_superlist.append([['Ink'],[[800,1,1]]])
        #formulation_superlist.append([['Ink'],[[800,2,1]]])

        #formulation_superlist.append([['Ink'],[[750,1,1]]])
        #formulation_superlist.append([['Ink'],[[750,2,1]]])

        #formulation_superlist.append([['Ink'],[['skip',1,'skip']]])
        #formulation_superlist.append([['Ink'],[['skip',2,'skip']]])
        #formulation_superlist.append([['Ink'],[['skip',3,'skip']]])
        #formulation_superlist.append([['Ink'],[['skip',4,'skip']]])

        #formulation_superlist.append([['Ink'],[[200,'skip',18.75]]])
        #formulation_superlist.append([['Ink'],[[200,'skip',25]]])
        #formulation_superlist.append([['Ink'],[[200,'skip',31.25]]])
        #formulation_superlist.append([['Ink'],[[300,'skip',12.5]]])
        #formulation_superlist.append([['Ink'],[[300,'skip',18.75]]])
        #formulation_superlist.append([['Ink'],[[300,'skip',25]]])

        #formulation_superlist.append([['_30_Pbinder_S.FM1'],[[0,'skip','skip']]])
        #formulation_superlist.append([['_700ms_S.FM1'],[[0,'skip','skip']]])
        #formulation_superlist.append([['30_50-50_S.FM1'],[[0,'skip','skip']]])
        #formulation_superlist.append([['60_6431_S.FM1'],[[0,'skip','skip']]])
        
       # formulation_superlist.append([["80mS"],[[' ','skip','skip']]])
       # formulation_superlist.append([["120mS"],[[' ','skip','skip']]])
       # formulation_superlist.append([["80mS"],[['skip','skip','_']]])

     #   formulation_superlist.append([["80mS"],[['skip','skip','skip']]])

     
        #formulation_superlist.append([["100mS"],[['EV0','skip','skip']]])
        #formulation_superlist.append([["100mS"],[['EV0','Nike','skip']]])
        #formulation_superlist.append([["100mS"],[['EV0','Adidas','skip']]])
        #formulation_superlist.append([["120mS"],[['EV0','skip','skip']]])
        #formulation_superlist.append([["120mS"],[['EV0','Nike','skip']]])
        #formulation_superlist.append([["120mS"],[['EV0','Adidas','skip']]])
        #formulation_superlist.append([["150mS"],[['EV0','skip','skip']]])
        #formulation_superlist.append([["150mS"],[['EV0','Nike','skip']]])
        #formulation_superlist.append([["150mS"],[['EV0','Adidas','skip']]])

        #formulation_superlist.append([["100mS"],[['EV-1','skip','skip']]])
        #formulation_superlist.append([["100mS"],[['EV-1','Nike','skip']]])
        #formulation_superlist.append([["100mS"],[['EV-1','Adidas','skip']]])
       # formulation_superlist.append([["120mS"],[['EV-1','skip','skip']]])
        #formulation_superlist.append([["120mS"],[['EV-1','Nike','skip']]])
        #formulation_superlist.append([["120mS"],[['EV-1','Adidas','skip']]])
        #formulation_superlist.append([["150mS"],[['EV-1','skip','skip']]])
        #formulation_superlist.append([["150mS"],[['EV-1','Nike','skip']]])
        #formulation_superlist.append([["150mS"],[['EV-1','Adidas','skip']]])

        #formulation_superlist.append([["120mS"],[['skip','skip','skip']]])
        #formulation_superlist.append([["120mS"],[['skip','Nike','skip']]])
        #formulation_superlist.append([["120mS"],[['skip','Adidas','skip']]])

     #   formulation_superlist.append([["200mS"],[['0.75 inch','skip','skip']]])
    #    formulation_superlist.append([["200mS"],[['0.5 inch','skip','skip']]])
    #    formulation_superlist.append([["200mS"],[['0.34 inch','skip','skip']]])

    ##    formulation_superlist.append([["150mS"],[['0.75 inch','skip','skip']]])
    #    formulation_superlist.append([["150mS"],[['0.50 inch','skip','skip']]])
    #    formulation_superlist.append([["150mS"],[['0.34 inch','skip','skip']]])


    #    formulation_superlist.append([["120mS"],[['skip','skip','skip']]])
       ## formulation_superlist.append([["80mS"],[['skip','skip','skip']]])
      #  formulation_superlist.append([["150mS"],[['skip','skip','skip']]])
       # formulation_superlist.append([["Print"],[['skip','skip','skip']]])
        formulation_superlist.append([["skip"],[['skip','skip','skip']]])

        #[FORMULATION,RECIPE]->[FORMULATION,[INK,BINDER,SOLVENT]]->[FORMULATION,[[INK,BINDER,SOLVENT],[INK2,BINDER2,SOLVENT2]]]

        mod_list=[]
        #mod_list.append('W0')
        #mod_list.append('W1')
        #mod_list.append('W2')
        #mod_list.append('W3')
        #mod_list.append('W4')
        #mod_list.append('D0')
        #mod_list.append('D1')
        #mod_list.append('D2')
        #mod_list.append('D3')
        #mod_list.append('CG')
        #mod_list.append('DW0')
        #mod_list.append('DW1')
        #mod_list.append('DW2')
        #mod_list.append('WD0')
        #mod_list.append('WD1')
        #mod_list.append('WD2')
        #mod_list.append('PB0')
        #mod_list.append('PB15')
        #mod_list.append('PB22')
        #mod_list.append('PB25')
        #mod_list.append('PB27')
        #mod_list.append('PB39')
        #mod_list.append('M0')
        mod_list=['skip']

       # mod_list.append("AP1")
        #mod_list.append("AP3")
        #mod_list.append("AP5")
        #mod_list.append("AP8")
        #mod_list.append("AP9")

        #mod_list.append("WSB")
        #mod_list.append("WWB")


        #mod_list.append("BSB")
        #mod_list.append("BWS")
        #mod_list.append("bsw")
        #mod_list.append('1coat')
       # mod_list.append('2coat')

        #mod_list.append('1')
        #mod_list.append('2')
        #mod_list.append('3')

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
                       # try:
                        if True:
                            if test_index==0:
                                for ROI in ROI_list:
                                    df_input=copy.copy(df)
                                    df_blank_input=copy.copy(df_blank)
                                    formulation=formulation_list[0][0]
                                    df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                                    df_fin=fun.cg_combine_print_blank(df_input,df_blank)
                                    print df_fin
                                    df_fin.head()
                                    [x_train,y_train]=fun.logistic_regression_prep_cg(df_input,df_blank_input,ROI,ROI_max)
                                    ROI_avg_result.append(y_train.Mark.mean())
                                    if test_index==0:
                                        result=fun.logistic_regression_model(x_train,y_train)
                                
                                    ROI_result.append(result[0])
                                print filename,
                                print ",",
                                print formulation_list[0][0],
                                print ",",
                                print recipe[0],
                                print ",",
                                print recipe[1],
                                print ",",
                                print recipe[2],
                                print ",",
                                print mod,
                                print ",",
                                print ROI_result[0],
                                print ",",
                                print ROI_result[1],
                                print ",",
                                print ROI_result[2]
                            if test_index==1 or test_index==4 or test_index==5 or test_index==6:
                                df_input=copy.copy(df)
                                df_blank_input=copy.copy(df_blank)
                                formulation=formulation_list[0][0]

                                
                               # print df_blank.head()
                               # print df_input.head()
                                df_input=fun.path_filter(df_input,path)
                                df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                                #print recipe
                                #print df_blank.iloc[:,0:5]
                                #if recipe[1]==3:
                                #    black_recipe=1
                                #elif recipe[1]==4:
                                #    blank_recipe=2
                                if test_index==5:
                                    df_blank_input=fun.cg_dataframe_filter(df_blank_input,'Blank',['skip',recipe[1],'skip'],mod)
                                #df_blank_input=fun.cg_dataframe_filter(df_blank_input,'Blank',['skip',blank_recipe,'skip'],mod)
                                #print df_blank.iloc[:,0:5]
                                df_input=fun.arbitrary_exclude(df_input,'formulation','blank')
                                df_fin=fun.cg_combine_print_blank(df_input,df_blank_input)
                                #print df_fin
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
                                print ",",
                                print formulation_list[0][0],
                                print ",",
                                print str(recipe[0])+"-"+str(recipe[1])+"-"+str(recipe[2]),
                                print ",",
                                #print recipe[1],
                                #print ",",
                                #print recipe[2],
                                #print ",",
                                print mod,
                                print ",",
                                print result[0][0],
                                print ",",
                                print result[0][1],
                                print ",",
                                print result[0][2],
                                print ",",
                                print result[0][3],
                                print ",",
                                print result[0][4],
                                print ",",
                                print result[1],
                                print ",",
                                print result[2],
                                print ",",
                                print result[3]
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
                                print ",",
                                print formulation_list[0][0],
                                print ",",
                                print str(recipe[0])+"-"+str(recipe[1])+"-"+str(recipe[2]),
                                print ",",
                                print mod,
                                print ",",
                                print result[0],
                                print ",",
                                print result[1],
                                print ",",
                                print result[2],
                                print ",",
                                print result[3],
                                print ",",
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
                                print ",",
                                print formulation_list[0][0],
                                print ",",
                                print str(recipe[0])+"-"+str(recipe[1])+"-"+str(recipe[2]),
                                print ",",
                                print mod,
                                print ",",
                                print result[0][0],
                                print ",",
                                print result[0][1],
                                print ",",
                                print result[0][2],
                                print ",",
                                print result[0][3],
                                print ",",
                                print result[1][0],
                                print ",",
                                print result[1][1],
                                print ",",
                                print result[1][2],
                                print ",",
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
                                        print ",",
                                        print best_sen,
                                        print ",",
                                        print best_spec,
                                        print ",",
                                        print count_blank,
                                        print ",",
                                        print count_print,
                                        print ",",
                                        print best_red,
                                        print ",",
                                        print path,
                                        print ",",
                                        print formulation,
                                        print ",",
                                        print recipe,
                                        print ",",
                                        print mod,
                                        print ",",
                                        print best_guess_mat

                                        if test_index==15:
                                            max_J,best_sen,best_spec,count_blank,count_print,best_red,best_guess_mat,fail_pic_list=fun.pk_redundancy_checker(df_test,best_guess_mat,best_red,[1])
                                            print max_J,
                                            print ",",
                                            print best_sen,
                                            print ",",
                                            print best_spec,
                                            print ",",
                                            print count_blank,
                                            print ",",
                                            print count_print,
                                            print ",",
                                            print best_red,
                                            print ",",
                                            print path,
                                            print ",",
                                            print formulation,
                                            print ",",
                                            print recipe,
                                            print ",",
                                            print mod,
                                            print ",",
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
                                    print ",",
                                    print best_sen,
                                    print ",",
                                    print best_spec,
                                    print ",",
                                    print count_blank,
                                    print ",",
                                    print count_print,
                                    print ",",
                                    print best_red,
                                    print ",",
                                    print path,
                                    print ",",
                                    print formulation,
                                    print ",",
                                    print recipe,
                                    print ",",
                                    print mod,
                                    print ",",
                                    print best_guess_mat,
                                    #print ",",
                                    #print fail_pic_list

#                        except:
 #                           skipped=True
                            #print 'SKIPPED: ',
                            #print filename,
                            #print ",",
                            #print formulation_list[0][0],
                            #print ",",
                            #print recipe[0],
                            #print ",",
                            #print recipe[1],
                            #print ",",
                            #print recipe[2],
                            #print ",",
                            #print mod
                        
                            





if(0): #csv import and sort
    csv_file="G://Developer//csvExports//InitialCP Sweep//Combo_data.csv"
    csv_file="G://Developer//csvExports//171124 Distance Test//Combo_data.csv"
    csv_file="G://Developer//csvExports//171124 TriPic Test//Combo_data.csv"
    fun.import_and_sort_csv(csv_file,1000,0)

if(0): #make a list
    numbers=[301,303,307,309,313,324,326,339,343,348,353]
    start_number_of_digits=1
    print "{",
    #for iter in range(start_number_of_digits,len(numbers)+1):
    for iter in range(start_number_of_digits,2):
        for val in it.combinations(numbers,iter):
            x_cols=list(val)
            x_cols.extend([2,10,17,102])
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
    #        print str(val2)+",",