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


    pattern   = "*.jpg"

    for dir,_,_ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir,pattern))) 
    for file in files:
        print file,
        try:
            UC=fun.print_exif_DT(file)
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
    test_index=4
    files=[]
    start_dir='C://Users//gttho//Documents//Visual Studio 2017//Projects//PythonApplication1//PythonApplication1//csv_import'
    pattern   = "*.csv"
    print 'prog1'
    for dir,_,_ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir,pattern))) 
    for file in files:
        print 'prog2'
        filename=file.rpartition("/")[2]
        #Import csv to dataframe only once
        df=pd.read_csv(file,header=0)
        
        #remove NAN values from df
        column_where_numbers_start=2;
        header_list=df.columns.values.tolist()
        #FIRST, CONVERT COLUMNS TO FLOATS
        for index in range(column_where_numbers_start,len(header_list)): #ONLY OPERATE ON COLUMNS PAST COLUMN 5
            df[header_list[index]] = df[header_list[index]].apply(pd.to_numeric,errors='coerce')
        #THEN REMOVE NAN VALUES
        df=df.dropna(subset = header_list[column_where_numbers_start:])

        #Make Dataframe of Blanks
        df_blank=df.loc[df['Formulation'].str.contains('blank|Blank')]

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
        formulation_superlist.append([['Ink'],[[750,1,0]]])
        formulation_superlist.append([['Ink'],[[750,2,0]]])

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
        #mod_list.append('M0')
        #mod_list=['skip']

        mod_list.append("WSB")
        mod_list.append("WWB")
        #mod_list.append('1coat')
       # mod_list.append('2coat')

        #mod_list.append('1')
        #mod_list.append('2')
        #mod_list.append('3')

        ROI_list=range(3)
        #x_col_list=[] #these should be COLUMN HEADERS
        #x_col_list.append()

        ROI_max=10;

        for formulation_list in formulation_superlist: #elements with the same formulation are clustered together
            for recipe in formulation_list[1]: #elements with different recipes are run
                for mod in mod_list:
                    ROI_result=[]
                    ROI_avg_result=[]
                    #try:
                    if True:
                        if test_index==0:
                            for ROI in ROI_list:
                                df_input=copy.copy(df)
                                df_blank_input=copy.copy(df_blank)
                                formulation=formulation_list[0][0]
                                df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                                df_fin=fun.cg_combine_print_blank(df_input,df_blank)
                                #print df_fin
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
                        if test_index==1 or test_index==4:
                            df_input=copy.copy(df)
                            df_blank_input=copy.copy(df_blank)
                            formulation=formulation_list[0][0]
                            df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                            df_fin=fun.cg_combine_print_blank(df_input,df_blank)
                            #print df_fin
                            if test_index==1:
                                result=fun.cg_redundancy_modeler(df_fin)
                            elif test_index==4:
                                result=fun.cg_redundancy_modeler_v2(df_fin)
                                

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
                            print result[0],
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
                            df_input=fun.cg_dataframe_filter(df_input,formulation,recipe,mod)
                            df_fin=fun.cg_combine_print_blank(df_input,df_blank)

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


                            result=fun.cg_redundancy_tester(df_fin,best_roi_thresh,best_dec_thresh,best_redundancy,print_failures=False)
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
                            df_fin=fun.cg_combine_print_blank(df_input,df_blank)

                           

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
                    #except:
                     #   skipped=True
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


    #print result
    #for val in result[3]:
    #    for val2 in val:
    #        print str(val2)+",",