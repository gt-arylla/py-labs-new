#functions
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import copy
import itertools as it
import csv
import PIL.Image
import PIL.ExifTags
import pandas as pd
import statsmodels.discrete.discrete_model as sm
import statsmodels.tools.tools as sm_tool
from operator import itemgetter
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import shutil
import os

#def df_to_dict(input_df):
#    df=copy.copy(input_df)

#    rough_dict=df.to_dict()

#    print rough_dict

#    clean_dict={}
#    for key in rough_dict:
#        clean_dict[key]=rough_dict[key][0]

#    return clean_dict 

def dropNAN(input_df,column_where_numbers_start):
    df=input_df
    #remove NAN values from df
    #column_where_numbers_start=8;
    header_list=df.columns.values.tolist()
    #FIRST, CONVERT COLUMNS TO FLOATS
    for index in range(column_where_numbers_start,len(header_list)): #ONLY OPERATE ON COLUMNS PAST COLUMN x
        df[header_list[index]] = df[header_list[index]].apply(pd.to_numeric,errors='coerce')
    #THEN REMOVE NAN VALUES
    df=df.dropna(subset = header_list[column_where_numbers_start:])
    return df

#converts nan to binary where nan values are 0 and non-nan values are 1
def nanTObin(input_mat,flipper=False):
    output_mat = np.ones(input_mat.shape)
    if not flipper:
        for i in np.arange(input_mat.shape[0]):
            for j in np.arange(input_mat.shape[1]):
                if np.isnan(input_mat[i,j]):
                    output_mat[i,j] = 0
    else:
        output_mat[:] = np.nan
        for i in np.arange(input_mat.shape[0]):
            for j in np.arange(input_mat.shape[1]):
                if not input_mat[i,j] == 0:
                    output_mat[i,j] = 0
    return output_mat

#makes a radially symmetrical gaussian filter based on input image dimentions
def gauss_maker(fig_rows,fig_cols,sigma_perc,anchor_perc_rows,anchor_perc_cols,max_val=1,flipper=False):
    #initialize sigma and k
    sigma = np.sqrt(fig_rows ** 2 + fig_cols ** 2) * sigma_perc
    
    k = int(max([fig_rows,fig_cols])) * 2
    sigma_fix = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    print str(sigma) + ',' + str(sigma_fix)
    #Calculate kernel
    kernel_1d = cv2.getGaussianKernel(k,sigma)
    kernel_1d = kernel_1d / max(kernel_1d) * max_val
    filler_fig = np.ones([fig_rows,fig_cols])

    anchor_rows = int(fig_rows * anchor_perc_rows)
    anchor_cols = int(fig_cols * anchor_perc_cols)

    for j in np.arange(fig_cols):
        for i in np.arange(fig_rows):
            anchor_diff_row = i - anchor_rows
            anchor_diff_col = j - anchor_cols
            kernel_loc_row = int(np.ceil(k / float(2)) + anchor_diff_row)
            kernel_loc_col = int(np.ceil(k / float(2)) + anchor_diff_col)
            # print kernel_loc
            filler_fig[i,j] = filler_fig[i,j] * kernel_1d[kernel_loc_row] * kernel_1d[kernel_loc_col]
    if flipper:
        final_fig = -(filler_fig - max_val)
        filler_fig = final_fig
    return filler_fig

#automatically makes subplots from a list of mats
def plotter(input_list):
    numberOFplots = len(input_list)
    for i in np.arange(numberOFplots):
        plt.subplot(1,numberOFplots,i + 1)
        plt.imshow(input_list[i])
    plt.show()
    return

def black_white_modeler(dataframe_input,white_index,black_count,test_type=[1],bin_count=55,blackavg_count=1):
    #code will accept a dataframe input.  It'll then find the optimal threshold for each combination of bin and black/white, according to test_type

    #The format for column headers is 'whitexbiny' or 'blackxbiny' where x is the color index and y is the bin index.
    #for each test, the result will be saved in a dataframe

    mark_list=list(dataframe_input["mark"])
    count_print=np.sum(mark_list)
    count_blank=len(mark_list)-count_print

    thresh_iterators=100

    data_dict={"test_name":[],"thresh":[],"j_abs":[],"j":[],"sen":[],"spec":[],"n_p":[],"n_b":[],"white_index":[],"white_bin":[],"black_index":[],"black_bin":[]}

    #print "Analysis Started"

    #part is simply running through all the white rings
    if 1 in test_type:
        for bin_index in range(bin_count):
            data_index="white"+str(white_index)+"bin"+str(bin_index)
            data_input=list(dataframe_input[data_index])
            thresh,J_abs,J,sen,spec=threshold_finder(data_input,mark_list,thresh_iterators)
            #print [thresh,J_abs,J,sen,spec]
            test_name=data_index
            data_dict["test_name"].append(test_name)
            data_dict["thresh"].append(thresh)
            data_dict["j_abs"].append(J_abs)
            data_dict["j"].append(J)
            data_dict["sen"].append(sen)
            data_dict["spec"].append(spec)
            data_dict["n_p"].append(count_print)
            data_dict["n_b"].append(count_blank)
            data_dict["white_index"].append(white_index)
            data_dict["white_bin"].append(bin_index)
            data_dict["black_index"].append('notused')
            data_dict["black_bin"].append('notused')


            data=data.append(adder_df)

    #print "Part 1 Done"

    ##part 2 is finding the parallel difference between all the white and black columns
    if 2 in test_type:
        for black_index in range(black_count):
            for bin_index in range(bin_count):
                pos_data_index="white"+str(white_index)+"bin"+str(bin_index)
                pos_data=list(dataframe_input[pos_data_index])
                neg_data_index="black"+str(black_index)+"bin"+str(bin_index)
                neg_data=list(dataframe_input[neg_data_index])
                diff_data=list(np.array(pos_data)-np.array(neg_data))
                thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                test_name=pos_data_index+"_minus_"+neg_data_index
                data_dict["test_name"].append(test_name)
                data_dict["thresh"].append(thresh)
                data_dict["j_abs"].append(J_abs)
                data_dict["j"].append(J)
                data_dict["sen"].append(sen)
                data_dict["spec"].append(spec)
                data_dict["n_p"].append(count_print)
                data_dict["n_b"].append(count_blank)
                data_dict["white_index"].append(white_index)
                data_dict["white_bin"].append(bin_index)
                data_dict["black_index"].append(black_index)
                data_dict["black_bin"].append(bin_index)

    ##part 3 is finding the parallel difference between all the white and averaged black column
    if 3 in test_type:
        for blackavg_index in range(blackavg_count):
            for bin_index in range(bin_count):
                pos_data_index="white"+str(white_index)+"bin"+str(bin_index)
                pos_data=list(dataframe_input[pos_data_index])
                neg_data_index="blackavg"+str(blackavg_index)+"bin"+str(bin_index)
                neg_data=list(dataframe_input[neg_data_index])
                diff_data=list(np.array(pos_data)-np.array(neg_data))
                thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                test_name=pos_data_index+"_minus_"+neg_data_index
                data_dict["test_name"].append(test_name)
                data_dict["thresh"].append(thresh)
                data_dict["j_abs"].append(J_abs)
                data_dict["j"].append(J)
                data_dict["sen"].append(sen)
                data_dict["spec"].append(spec)
                data_dict["n_p"].append(count_print)
                data_dict["n_b"].append(count_blank)
                data_dict["white_index"].append(white_index)
                data_dict["white_bin"].append(bin_index)
                data_dict["black_index"].append(-blackavg_index)
                data_dict["black_bin"].append(bin_index)

    ##part 4 is finding the difference between all combinations of the white and black columns
    if 4 in test_type:
        for black_index in range(black_count):
            for bin_index_white in range(bin_count):
                for bin_index_black in range(bin_count):
                    pos_data_index="white"+str(white_index)+"bin"+str(bin_index_white)
                    pos_data=list(dataframe_input[pos_data_index])
                    neg_data_index="black"+str(black_index)+"bin"+str(bin_index_black)
                    neg_data=list(dataframe_input[neg_data_index])
                    diff_data=list(np.array(pos_data)-np.array(neg_data))
                    thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                    test_name=pos_data_index+"_minus_"+neg_data_index
                    data_dict["test_name"].append(test_name)
                    data_dict["thresh"].append(thresh)
                    data_dict["j_abs"].append(J_abs)
                    data_dict["j"].append(J)
                    data_dict["sen"].append(sen)
                    data_dict["spec"].append(spec)
                    data_dict["n_p"].append(count_print)
                    data_dict["n_b"].append(count_blank)
                    data_dict["white_index"].append(white_index)
                    data_dict["white_bin"].append(bin_index_white)
                    data_dict["black_index"].append(black_index)
                    data_dict["black_bin"].append(bin_index_black)

    ##part 5 is finding the difference between all combinations of the white and averaged black columns
    if 5 in test_type:
        for blackavg_index in range(blackavg_count):
            for bin_index_white in range(bin_count):
                for bin_index_black in range(bin_count):
                    pos_data_index="white"+str(white_index)+"bin"+str(bin_index_white)
                    pos_data=list(dataframe_input[pos_data_index])
                    neg_data_index="blackavg"+str(blackavg_index)+"bin"+str(bin_index_black)
                    neg_data=list(dataframe_input[neg_data_index])
                    diff_data=list(np.array(pos_data)-np.array(neg_data))
                    thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                    test_name=pos_data_index+"_minus_"+neg_data_index
                    data_dict["test_name"].append(test_name)
                    data_dict["thresh"].append(thresh)
                    data_dict["j_abs"].append(J_abs)
                    data_dict["j"].append(J)
                    data_dict["sen"].append(sen)
                    data_dict["spec"].append(spec)
                    data_dict["n_p"].append(count_print)
                    data_dict["n_b"].append(count_blank)
                    data_dict["white_index"].append(white_index)
                    data_dict["white_bin"].append(bin_index_white)
                    data_dict["black_index"].append(-blackavg_index)
                    data_dict["black_bin"].append(bin_index_black)

        ##part 6 is finding the difference between a restricted set of bins, all combinations of the white and black columns
    if 6 in test_type:
        bin_subset=[9,8,18,7,17,26,16,6,25,33,24,32,5,15,39,4,14,31,38,44,23]
        #bin_subset=[9,8,18,7,17,26,16,6,25,33]
        for black_index in range(black_count):
            for bin_index_white in bin_subset:
                for bin_index_black in bin_subset:
                    pos_data_index="white"+str(white_index)+"bin"+str(bin_index_white)
                    pos_data=list(dataframe_input[pos_data_index])
                    neg_data_index="black"+str(black_index)+"bin"+str(bin_index_black)
                    neg_data=list(dataframe_input[neg_data_index])
                    diff_data=list(np.array(pos_data)-np.array(neg_data))
                    thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                    test_name=pos_data_index+"_minus_"+neg_data_index
                    data_dict["test_name"].append(test_name)
                    data_dict["thresh"].append(thresh)
                    data_dict["j_abs"].append(J_abs)
                    data_dict["j"].append(J)
                    data_dict["sen"].append(sen)
                    data_dict["spec"].append(spec)
                    data_dict["n_p"].append(count_print)
                    data_dict["n_b"].append(count_blank)
                    data_dict["white_index"].append(white_index)
                    data_dict["white_bin"].append(bin_index_white)
                    data_dict["black_index"].append(black_index)
                    data_dict["black_bin"].append(bin_index_black)

    data=pd.DataFrame(data_dict)

    #Reset data index
    data=data.reset_index(drop=True)

   
   # print data

    # Filter out nan values
    df=dropNAN(data,0)

    # Determine maximum conditions
    data_max=data.ix[data['j_abs'].idxmax()]   
    data_max_dict=data_max.to_dict()
   # data_max=data_max.reset_index(drop=True)
    #result_max_list=list(result_max)

    #print data_max;

    return data,data_max_dict

#def black_white_tester(dataframe_in,parameter_dict):

#    df=copy.copy(df)

#    #Reset data index
#    df=df.reset_index(drop=True)

#    mark_list=list(df["mark"])
#    count_print=np.sum(mark_list)
#    count_blank=len(mark_list)-count_print

#    TP=0
#    TN=0
#    FP=0
#    FN=0
#    for df_row in range(df.shape[0]):
#        sum=0
#        for i in range(len(parameter_dict['white_index'])):
#            white_ID="white"+str(int(parameter_dict['white_index'][i]))+"bin"+str(int(parameter_dict['white_bin'][i]))
#            black_ID="black"+str(int(parameter_dict['black_index'][i]))+"bin"+str(int(parameter_dict['black_bin'][i]))
#            white_value=df.at[df_row,white_ID]
#            black_value=df.at[df_row,black_ID]
#        sum_list.append(sum)
#        if sum>red_thresh:
#            guess=1
#        else:
#            guess=0
#        mark=df.at[df_row,'mark']
#        #print "Sum: "+str(sum)
#        #print "RedThresh: "+str(red_thresh)
#        #print "Guess: "+str(guess)
#        #print "Mark: "+str(mark)
#        if mark==1:
#            if guess==1:
#                TP=TP+1
#            elif guess==0:
#                FN=FN+1
#        elif mark==0:
#            if guess==0:
#                TN=TN+1
#            elif guess==1:
#                FP=FP+1
#        #print [TP,TN,FP,FN]
#        #print ""
#    if not TP+FN==count_print:
#        raise ("Loop print count and total #print count do not match")
#    if not TN+FP==count_blank:
#        raise ("Loop blank count and total blank count do not match")
#    sensitivity=float(TP)/float(TP+FN)
#    specificity=float(TN)/float(TN+FP)
#    J=adapted_J(sensitivity,specificity)


    

def black_white_redundancy(dataframe_input,white_count,black_count,test_type=[1],bin_count=55):
    #build a list of dataframes for all white ROIs
    white_list_dict=[]
    ROI_thresh_dict={}
  #  print white_count
    for white_index in range(white_count):
        #print white_index
        temp_df=black_white_modeler(dataframe_input,white_index,black_count,test_type,bin_count)[1]
        #print white_index
       # print temp_df
        #print temp_df
        white_list_dict.append(temp_df)
        ROI_thresh_dict["roi"+str(white_index)]=temp_df["thresh"]

    #reset input df index so that it can be iterated through
    df=copy.copy(dataframe_input)
    df=df.reset_index(drop=True)
    

    #iterate through df, making a nested list of accuracy binaries
    guess_list=[]
    ROI_threshes={}
    for df_row in range(df.shape[0]):
        temp_list=[]
        for ROI_index in range(white_count):
            ROI_index = int(ROI_index)
            white_column='white'+str(int(white_list_dict[ROI_index]["white_index"]))+'bin'+str(int(white_list_dict[ROI_index]["white_bin"]))
            black_column='black'+str(int(white_list_dict[ROI_index]["black_index"]))+'bin'+str(int(white_list_dict[ROI_index]["black_bin"]))
            #print ROI_index,
            #print "  ",
            #print white_column,
            #print "  ",
            #print black_column
            threshold_value=ROI_thresh_dict["roi"+str(ROI_index)]
            diff=df.at[df_row,white_column]-df.at[df_row,black_column]

            if diff>threshold_value:
                temp_list.append(1)
            else:
                temp_list.append(0)
        guess_list.append(temp_list)
    
    #prep data to check against later
    mark_list=list(dataframe_input["mark"])
    count_print=np.sum(mark_list)
    count_blank=len(mark_list)-count_print

    #iterate through redundancy levels, always considering ALL the ROIs (in the future, I can loop this to only consider some ROIs)
   
   # active_ROIs=[0]
    best_J=0
    best_sen=0
    best_spec=0
    best_active_ROIs=[]
    best_red_thresh=0
    available_ROIs=range(white_count)
    sum_list_final=[]
    min_ROI_count_sweep=1

    for ROI_count in range(min_ROI_count_sweep,white_count+1):
        for active_ROIs in list(it.combinations(range(0,len(available_ROIs)),ROI_count)):


    


            for red_thresh in np.linspace(0.5,len(active_ROIs)-0.5,len(active_ROIs)):
                sum_list=[]
                TP=0
                TN=0
                FP=0
                FN=0
                for df_row in range(df.shape[0]):
                    sum=0
                    for ROI in active_ROIs:
                        sum=sum+guess_list[df_row][ROI]
                    sum_list.append(sum)
                    if sum>red_thresh:
                        guess=1
                    else:
                        guess=0
                    mark=df.at[df_row,'mark']
                    #print "Sum: "+str(sum)
                    #print "RedThresh: "+str(red_thresh)
                    #print "Guess: "+str(guess)
                    #print "Mark: "+str(mark)
                    if mark==1:
                        if guess==1:
                            TP=TP+1
                        elif guess==0:
                            FN=FN+1
                    elif mark==0:
                        if guess==0:
                            TN=TN+1
                        elif guess==1:
                            FP=FP+1
                    #print [TP,TN,FP,FN]
                    #print ""
                if not TP+FN==count_print:
                    raise ("Loop print count and total #print count do not match")
                if not TN+FP==count_blank:
                    raise ("Loop blank count and total blank count do not match")
                sensitivity=float(TP)/float(TP+FN)
                specificity=float(TN)/float(TN+FP)
                J=adapted_J(sensitivity,specificity)
                #print "J: "+str(J)
                #print "Sen: "+str(sensitivity)
                #print "Spec: "+str(specificity)
                if J>best_J:
                    #print "################## NEW J ##########"
                    best_J=J
                    best_sen=sensitivity
                    best_spec=specificity
                    best_active_ROIs=active_ROIs
                    best_red_thresh=red_thresh
                    sum_list_final=sum_list
                    #print "J: "+str(J)
                    #print "Sen: "+str(sensitivity)
                    #print "Spec: "+str(specificity)

    save_failed_images(df,mark_list,sum_list_final,best_red_thresh)

    output_dict={"j":best_J,"sen":best_sen,"spec":best_spec,"active_rois":best_active_ROIs,"red_thresh":best_red_thresh,"roi_thresh":ROI_thresh_dict,"optimal_rois":white_list_dict,"n_b":count_blank,"n_p":count_print}
    for key in ROI_thresh_dict:
        output_dict[key]=ROI_thresh_dict[key]


    return output_dict
    

#converts from bgr to cct
def cct(input_img):
    i_ext = cv2.cvtColor(input_img,cv2.COLOR_BGR2XYZ)
    #plt.imshow(i_ext)
    #plt.show()
    i_cct = np.zeros_like(i_ext[:,:,0],float)
    #i_cct.astype(float)
    for i in range(0,i_ext.shape[0]):
        for j in range(0,i_ext.shape[1]):
            X = float(i_ext[i,j,0])
            Y = float(i_ext[i,j,1])
            Z = float(i_ext[i,j,2])
            #print X
            #print Y
            #print Z
            breaker=False
            if not (X+Y+Z)==0:
                x = (X) / (X + Y + Z);
                y = (Y) / (X + Y + Z);
            else:
                breaker=True
            if not breaker:
                if not (0.1858 - y)==0:
                    n = (x - 0.3320) / (0.1858 - y);
                else:
                    breaker=True
           # print n
            if not breaker: 
                CCT = 449.0*pow(n, 3) + 3525.0*pow(n, 2) + 6823.3*n + 5520.33;
            #print CCT
            if CCT<20000 and CCT>0 and not breaker:
                i_cct[i,j] = CCT
    print type(i_cct[0,0])
    print i_cct
    return i_cct


def area_filter(input_img,min_area,max_area=0):
    contours, hierarchy = cv2.findContours(copy.copy(input_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print len(contours)
    contour_filter=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        #if not min_area==0:
        #    if area>min_area and area<max_area:
        #        contour_filter.append(cnt)
        if True:
            if area>min_area:
                contour_filter.append(cnt)
                #print 'append'
    print len(contour_filter)
    img_black=np.zeros_like(input_img)
    plt.imshow(img_black)
    plt.show()
    cv2.drawContours(img_black,contour_filter,-1,(255))
    plt.imshow(img_black)
    plt.show()
    
    return img_black

def multi_photo_model(csv_file,photo_number,test_type,unique_array,mark,confidence,mark_pred, printer=False):
     #count rows
    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        row_count = sum(1 for row in readCSV)

    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            col_count = len(row)
            break

    #Import Data
    if printer:
        print 'Import Data...'
    all_data = np.zeros(shape=(row_count - 1,col_count))
    row_count = 0
    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        counter = 0
        for row in readCSV:
            if counter > 0:
                try:
                    all_data[row_count,:] = row
                    row_count+=1
                except:
                    if printer:
                        print "Row Import Failed"
            else:
                headers = row
            counter+=1
    if printer:
        print all_data

    unique_vals = []
    for i in unique_array:
        unique_list = np.unique(all_data[:,i])
        unique_vals.append(unique_list)

    if printer:
        print unique_vals

    #Construct vector of unique values, saved as strings
    if printer:
        print 'Construct vector of unique values, saved as strings...'
    unique_vals_string = []
    for i in np.arange(len(unique_vals)):
        unique_vals_string.append([])
        for j in np.arange(len(unique_vals[i])):
            unique_vals_string[i].append(str(unique_vals[i][j]))

    #construct super vector with all possible combinations of unique vals
    if printer:
        print 'construct super vector with all possible combinations of unique vals...'
        print unique_vals_string
    super_list = [" ".join(i) for i in it.product(*unique_vals_string)]

    #for items in it.product(*unique_vals_string):
    #    super_list.append(items)
    
    if printer:
        print super_list

    #construct holding dictionary
    if printer:
        print 'construct holding dictionary...'
    values_dictionary = {}
    for i in np.arange(len(super_list)):
        values_dictionary[super_list[i]] = [[],[],[]]
    if printer:
        print values_dictionary.keys()
    #add data to holding dictionary
    if printer:
        print 'add data to holding dictionary...'
    for row in np.arange(row_count):
        key = ''
        for val in unique_array:
            key+=str(all_data[row,val]) + ' '
            
        key = key[:-1]
        #print key
        values_dictionary[key][0].append(all_data[row,mark])
        values_dictionary[key][1].append(all_data[row,confidence])
        values_dictionary[key][2].append(all_data[row,mark_pred])

    #plot data
    if printer:
        print 'model data...'
    TP=0;
    FP=0;
    TN=0;
    FN=0;
    super_counter=0
    for i in super_list:
        mark_data=values_dictionary[i][0]
        confidence_data=values_dictionary[i][1]
        mark_pred_data=values_dictionary[i][2]
        data_set_size=len(confidence_data)
        if data_set_size>0:
            #print "DS size: "+str(data_set_size)
            combo_generator=it.combinations(range(0,data_set_size),photo_number)
            #print "CG size: "+str(len(list(combo_generator)))
            #print list(combo_generator)
            #print mark_data
            counter=0
            #print 'reset counter'
            print_pred_counter=0
            for pic_set in combo_generator:
                confidence_mini_list=[]
                mark_pred_mini_list=[]
                for j in range(0,photo_number):
                    confidence_mini_list.append(confidence_data[pic_set[j]])
                    mark_pred_mini_list.append(mark_pred_data[pic_set[j]])
               # print sum(mark_pred_mini_list)
                #print counter
               # print print_pred_counter
                if test_type==0:#zero photo method
                    if sum(mark_pred_mini_list)>=-0.5:
                        print_pred_counter+=1
                if test_type==1:#one photo method
                    if sum(mark_pred_mini_list)>=0.5:
                        print_pred_counter+=1
                if test_type==2:#two photo method
                    if sum(mark_pred_mini_list)>=1.5:
                        print_pred_counter+=1
                if test_type==3:#three photo method
                    if sum(mark_pred_mini_list)>=2.5:
                        print_pred_counter+=1
                if test_type==4:#four photo method
                    if sum(mark_pred_mini_list)>=3.5:
                        print_pred_counter+=1
                if test_type==5:#five photo method
                    if sum(mark_pred_mini_list)>=3.5:
                        print_pred_counter+=1
               #if test_type==0:#one photo method
                #    if sum(mark_pred_mini_list)>=0.5:
                #        print_pred_counter+=1
                #if test_type==20:
                #    if sum(mark_pred_mini_list)==2:
                #        print_pred_counter+=1
                #if test_type==21:
                #    max_confidence=max(confidence_mini_list)
                #    for k in range(0,len(confidence_mini_list)):
                #        if confidence_mini_list[k]==max_confidence:
                #            pred_val_final=mark_pred_mini_list[k]
                #    if pred_val_final==1:
                #        print_pred_counter+=1
                #if test_type==30:#three pics, best two out of three of the mark_pred wins
                #    if sum(mark_pred_mini_list)>1.5:
                #        print_pred_counter+=1
                #if test_type==31:#three pics, best two out of three of the mark_pred wins
                #    if sum(mark_pred_mini_list)>2.5:
                #        print_pred_counter+=1
                #if test_type==32:
                #    max_confidence=max(confidence_mini_list)
                #    for k in range(0,len(confidence_mini_list)):
                #        if confidence_mini_list[k]==max_confidence:
                #            pred_val_final=mark_pred_mini_list[k]
                #    if pred_val_final==1:
                #        print_pred_counter+=1
                #if test_type==33:
                #    avg_pred_val=sum(mark_pred_mini_list)/float(len(mark_pred_mini_list))
                #    if avg_pred_val==0:
                #        null=1;
                #    elif avg_pred_val==1:
                #        print_pred_counter+=1
                #    else: #split into blank and print 
                #        print_vec=[]
                #        blank_vec=[]
                #        for k in range(0,len(mark_pred_mini_list)):
                #            if mark_pred_mini_list[k]==0:
                #                blank_vec.append(confidence_mini_list[k])
                #            elif mark_pred_mini_list[k]==1:
                #                print_vec.append(confidence_mini_list[k])
                #        blank_conf_avg=sum(blank_vec)/float(len(blank_vec))
                #        print_conf_avg=sum(print_vec)/float(len(print_vec))
                #        if print_conf_avg>blank_conf_avg:
                #            print_pred_counter+=1

                #if test_type==4:#three pics, best two out of three of the mark_pred wins
                #    if sum(mark_pred_mini_list)>1.5:
                #        print_pred_counter+=1
                #if test_type==40:#three pics, best two out of three of the mark_pred wins
                #    if sum(mark_pred_mini_list)>2.5:
                #        print_pred_counter+=1
                #if test_type==41:#three pics, best two out of three of the mark_pred wins
                #    if sum(mark_pred_mini_list)>3.5:
                #        print_pred_counter+=1
                #if test_type==42:#three pics, best two out of three of the mark_pred wins
                #    max_confidence=max(confidence_mini_list)
                #    for k in range(0,len(confidence_mini_list)):
                #        if confidence_mini_list[k]==max_confidence:
                #            pred_val_final=mark_pred_mini_list[k]
                #    if pred_val_final==1:
                #        print_pred_counter+=1
                #if test_type==43:
                #    avg_pred_val=sum(mark_pred_mini_list)/float(len(mark_pred_mini_list))
                #    if avg_pred_val==0:
                #        null=1;
                #    elif avg_pred_val==1:
                #        print_pred_counter+=1
                #    else: #split into blank and print 
                #        print_vec=[]
                #        blank_vec=[]
                #        for k in range(0,len(mark_pred_mini_list)):
                #            if mark_pred_mini_list[k]==0:
                #                blank_vec.append(confidence_mini_list[k])
                #            elif mark_pred_mini_list[k]==1:
                #                print_vec.append(confidence_mini_list[k])
                #        blank_conf_avg=sum(blank_vec)/float(len(blank_vec))
                #        print_conf_avg=sum(print_vec)/float(len(print_vec))
                #        if print_conf_avg>blank_conf_avg:
                #            print_pred_counter+=1
                counter+=1
                #print 'update counter'

            if mark_data[0]==0:
                FP+=print_pred_counter
                TN+=counter-print_pred_counter
            elif mark_data[0]==1:
                TP+=print_pred_counter
                FN+=counter-print_pred_counter
            if printer:
                print str(super_counter/float(len(super_list)))+" "+str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN)+" "+str(data_set_size)+" "+str(counter)
        super_counter+=1;
        

    true_positive_percentage=TP/float(TP+FN)
    false_positive_percentage=FP/float(FP+TN)
    truth=true_positive_percentage-false_positive_percentage
    if printer:
        print "Truth: "+str(truth)
        print "TPP: "+str(true_positive_percentage)
        print "FPP: "+str(false_positive_percentage)
    

    return truth, true_positive_percentage, false_positive_percentage

def logistic_regression(csv_file,x_cols,row_keep, printer,mark_pos=-1):
    data=pd.read_csv(csv_file, header=0)

    for keeper_list in row_keep:
        if len(keeper_list)==2:
            data=data.loc[data[keeper_list[0]].isin(keeper_list[1])]

    drop = range(data.shape[1])

    mark_loc=0
    drop=np.delete(drop,x_cols)
    if mark_pos==-1:
        mark_loc=len(x_cols)-1
    else:
        mark_loc=mark_pos
    data.drop(data.columns[drop], axis=1, inplace=True)
    print(data.head(10))
    data=data.dropna(axis=0,how='any')
  
    y = data.iloc[:,mark_loc]
    #Drop the 4th col for X data
    data.drop(data.columns[[mark_loc]], axis=1, inplace=True)
    X = data.iloc[:,:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.000, random_state=0)
    y_train.astype('int')
    y_test.astype('int')
    X_train.shape
    classifier = LogisticRegression(solver='newton-cg', random_state = 0,fit_intercept=True,class_weight=None)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_train)



    from sklearn.metrics import confusion_matrix
            
    confusion_matrix = confusion_matrix(y_train, y_pred)
    #print confusion_matrix
    tn=confusion_matrix[0,0]
    fp=confusion_matrix[0,1]
    tp=confusion_matrix[1,1]
    fn=confusion_matrix[1,0]
    tpr=tp/float(tp+fn)
    tnr=tn/float(tn+fp)
    fpr=fp/float(fp+tn)
    #truth=tpr-(1-tnr)
    truth=tpr-fpr
    
    if printer:
        print 'Truth: ' + str(truth)
        print 'TPP: '+str(tpr)
        print 'fpr: '+str(fpr)

        for row in range(0,len(X_train.index)):
            mark_perc= classifier.predict_proba(X_train.iloc[[row]])[0,1]
            LUX= X_train.iloc[[row]].values[0,14]
            print " ",
            mark= y_train.iloc[[row]].values[0]
            if mark_perc<0.5:
                pred_mark=0
                confidence=(0.5-mark_perc)*2
            else:
                pred_mark=1
                confidence=(mark_perc-0.5)*2
            if pred_mark==mark:
                accuracy=1
            else:
                accuracy=0
            multiple1=100
            multiple2=300
            if LUX<=1000:
                MultLux=np.floor((LUX + multiple1/2) / multiple1) * multiple1
            else:
                MultLux=np.floor((LUX + multiple2/2) / multiple2) * multiple2
            MultLux=int(MultLux)
            print str(mark_perc)+","+str(LUX)+","+str(MultLux)+","+str(mark)+","+str(pred_mark)+","+str(accuracy)+","+str(confidence)

        print(confusion_matrix)
        print y_train.mean()
        print classifier.coef_
        print classifier.intercept_

        print(str(data.shape[0])+', {:.5f}'.format(classifier.score(X_train, y_train)))+","+str(truth)+ "," + str(tpr) + "," + str(fpr) + "," + str(tp) + "," + str(fn) + "," + str(tn) + "," + str(fp)
    return [truth,tpr,fpr,classifier.coef_,classifier.intercept_]

def logistic_regression_prep(csv_file,x_cols,row_keep=[[0]], tst_size=0.000,mark_pos=-1,dataframe_checker=False,dataframe_input=[]):
    if dataframe_checker:
        data=dataframe_input
    else:
        data=pd.read_csv(csv_file, header=0)

    data = data[data.applymap(np.isreal).any(1)]

    for keeper_list in row_keep:
        if len(keeper_list)==2:
            data=data.loc[data[keeper_list[0]].isin(keeper_list[1])]

    drop = range(data.shape[1])

    mark_loc=0
    drop=np.delete(drop,x_cols)
    if mark_pos==-1:
        mark_loc=len(x_cols)-1
    else:
        mark_loc=mark_pos
    data.drop(data.columns[drop], axis=1, inplace=True)
    
    data=data.dropna(axis=0,how='any')
  
    y = data.iloc[:,mark_loc]
    #Drop the 4th col for X data
    data.drop(data.columns[[mark_loc]], axis=1, inplace=True)
    X = data.iloc[:,:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_size, random_state=0)
    y_train.astype('int')
    y_test.astype('int')
    
    return [X_train,y_train,X_test,y_test]

def adapted_J(sensitivity,specificity,n_print=0,n_CP=0):
    sen_weight=1
    spec_weight=3
    FN_weight=1
    CP_weight=2
    n_FN=(1-sensitivity)*n_print-n_CP
    if n_print==0:
        sen_mod=sensitivity
    else:
        sen_mod=1-(((n_FN*FN_weight)/float(n_print))+((n_CP*CP_weight)/float(n_print)))/float(FN_weight+CP_weight)

    J=2*(sen_weight*sensitivity+spec_weight*specificity)/float(sen_weight+spec_weight)-1
    #J=(sen_weight*sensitivity+spec_weight*specificity)/float(sen_weight+spec_weight)
    return J
   
def J_from_vectors(guess_list,input_mark):
    tp=0
    fp=0
    tn=0
    fn=0
    for data_index in range(len(guess_list)):
        mark_guess=guess_list[data_index]
        if input_mark[data_index] == 0:
            if mark_guess==0:
                tn+=1
            else:
                fp+=1
        else:
            if mark_guess==0:
                fn+=1
            else:
                tp+=1
    sen=float(tp)/float((tp+fn))
    spec=float(tn)/float((tn+fp))
    J=adapted_J(sen,spec)
    return J,sen,spec

def cg_dataframe_filter(dataframe_input,formulation,recipe,modification):
    df=copy.copy(dataframe_input)

    #KEEP ROWS THAT MATCH THE FORMULATION OR BLANK
    if not formulation=='skip':
        df=df.loc[df['Formulation'].str.contains(formulation)]
    
    #KEEP ROWS THAT MATCH THE RECIPE
    if not recipe[0]=='skip':
        df=df.loc[df['Ink'].isin([recipe[0]])]
    if not recipe[1]=='skip':
        df=df.loc[df['Binder'].isin([recipe[1]])]
    if not recipe[2]=='skip':
        df=df.loc[df['Solvent'].isin([recipe[2]])]

    #KEEP ROWS THAT MATCH THE MODIFICATION
    if not modification=='skip':
        #make lowercase and uppercase options
        lc_mod=modification.lower()
        uc_mod=modification.upper()
        df_uc=df.loc[df['Mod'].str.contains(lc_mod)]
        df_lc=df.loc[df['Mod'].str.contains(uc_mod)]
        df=pd.concat([df_uc,df_lc])
    return df

def cg_scan_range_finder(dataframe_input,roi_cols,roi_size):
    df=copy.copy(dataframe_input)
    #df_print,df_blank=[x for _, x in df.groupby(df['mark'] <0.5) ]

    #df_range=[df_blank,df_print]
    #max_min_export=[]
    #for df_index in range(len(df_range)):
    #    value_holder=[]
    #    for roi in range(roi_size):
    #        for scan in range(roi_cols):
    #            col_caller='roi_'+str(roi)+'_scan_'+str(scan)
    #            if df_index==0:
                    
    #                value_holder.append(df_range[df_index][col_caller].max())
    #            elif df_index==1:
    #                value_holder.append(df_range[df_index][col_caller].min())
    #    if df_index==0:
    #        max_min_export.append(np.max(value_holder))
    #    elif df_index==1:
    #        max_min_export.append(np.min(value_holder))
    max_min_export=[-5,15]
    return max_min_export

def cg_combine_print_blank(dataframe_print,dataframe_blank):
    dfp=copy.copy(dataframe_print)
    dfb=copy.copy(dataframe_blank)
    pLength = dfp.shape[0]
    bLength = dfb.shape[0]
    #print pLength
    #print bLength
    print_mark_series=pd.Series(np.ones(pLength))
    blank_mark_series=pd.Series(np.zeros(bLength))
    dfp=dfp.assign(mark=print_mark_series.values)
    dfb=dfb.assign(mark=blank_mark_series.values)
    df_fin=pd.concat([dfp,dfb],ignore_index=True)
    return df_fin

def cg_white_mean_finder(dataframe_input,scan_size=10):
    list_data_holder=cg_dataframe_to_list(dataframe_input,scan_size)
    roi_blank=[]
    roi_print=[]
    for row_index in range(len(list_data_holder)):
        sum_count=0
        roi_holder=[]
        for roi_index in range(3):
            white_holder=[]
            for scan_index in range(len(list_data_holder[row_index][0][roi_index])): 
                white_holder.append(list_data_holder[row_index][0][roi_index][scan_index])
            roi_mean=np.mean(white_holder)
            roi_holder.append(roi_mean)   
        if list_data_holder[row_index][1]==0:
            roi_blank.append(roi_holder)
        elif list_data_holder[row_index][1]==1:
            roi_print.append(roi_holder)
        else:
            print "************MARK ERROR********"
    roi0_blank=[]
    roi1_blank=[]
    roi2_blank=[]
    roi0_print=[]
    roi1_print=[]
    roi2_print=[]
    for roi_blank_single in roi_blank:
        roi0_blank.append(roi_blank_single[0])
        roi1_blank.append(roi_blank_single[1])
        roi2_blank.append(roi_blank_single[2])
    for roi_print_single in roi_print:
        roi0_print.append(roi_print_single[0])
        roi1_print.append(roi_print_single[1])
        roi2_print.append(roi_print_single[2])
    roi0_blank_mean=np.mean(roi0_blank)
    roi1_blank_mean=np.mean(roi1_blank)
    roi2_blank_mean=np.mean(roi2_blank)
    roi0_print_mean=np.mean(roi0_print)
    roi1_print_mean=np.mean(roi1_print)
    roi2_print_mean=np.mean(roi2_print)    
    roi_print_mean=np.mean([roi0_print_mean,roi1_print_mean,roi2_print_mean])
    roi_blank_mean=np.mean([roi0_blank_mean,roi1_blank_mean,roi2_blank_mean])
    return [[roi_print_mean,roi0_print_mean,roi1_print_mean,roi2_print_mean],[roi_blank_mean,roi0_blank_mean,roi1_blank_mean,roi2_blank_mean]]

def cg_redundancy_tester(dataframe_input,best_roi_thresh,best_dec_thresh,best_redundancy,print_failures=False,scan_size=10):
    list_data_holder=cg_dataframe_to_list(dataframe_input,scan_size)
    true_blank_guess_list=[]
    true_print_guess_list=[]
    for row_index in range(len(list_data_holder)):
        sum_count=0
        for roi_index in range(len(best_roi_thresh)):
            read_count=0
            for scan_index in range(len(list_data_holder[row_index][0][roi_index])): 
                if list_data_holder[row_index][0][roi_index][scan_index]>best_roi_thresh[roi_index]:
                    read_count+=1
            confidence_value=float(read_count)/float(len(list_data_holder[row_index][0][roi_index]))
            if confidence_value>best_dec_thresh[roi_index]:
                sum_count+=1
        if sum_count>best_redundancy:
            guess_val=1
        else:
            guess_val=0
        if list_data_holder[row_index][1]==0:
            true_blank_guess_list.append(guess_val)
        elif list_data_holder[row_index][1]==1:
            true_print_guess_list.append(guess_val)
        else:
            print "************MARK ERROR********"
        if print_failures:
            if not guess_val==list_data_holder[row_index][1]:
                print dataframe_input.iloc[[row_index]]
    sensitivity=np.mean(true_print_guess_list)
    specificity=1-np.mean(true_blank_guess_list)
    J=sensitivity+specificity-1
    n_blank=len(true_blank_guess_list)
    n_print=len(true_print_guess_list)

    return [J, sensitivity, specificity, n_print, n_blank]

def cg_redundancy_tester_detail(dataframe_input,best_roi_thresh,best_dec_thresh,best_redundancy,scan_size=10,print_failures=False): #Prints out original dataframe, as well as binary guess for each roi, and sucess/failure per guess
    #Reset row index of dataframe so that data can be grabbed easily
    dataframe_input=dataframe_input.reset_index(drop=True)
    #Convert to list for speed
    list_data_holder=cg_dataframe_to_list(dataframe_input,scan_size)
    true_blank_guess_list=[]
    true_print_guess_list=[]
    roi_guess_list=[[],[],[]] #only built for 3 rois
    roi_accuracy_list=[[],[],[]]
    for row_index in range(len(list_data_holder)):
        #for header_val in ['Formulation','Mod','Ink','Binder','Solvent','DateTime','mark']:
        #    print dataframe_input[header_val][row_index],
        #    print ',',

        sum_count=0
        for roi_index in range(len(best_roi_thresh)):
            read_count=0
            for scan_index in range(len(list_data_holder[row_index][0][roi_index])): 
                if list_data_holder[row_index][0][roi_index][scan_index]>best_roi_thresh[roi_index]:
                    read_count+=1
            confidence_value=float(read_count)/float(len(list_data_holder[row_index][0][roi_index]))
            if confidence_value>best_dec_thresh[roi_index]:
                sum_count+=1
                if list_data_holder[row_index][1]==1:
                    roi_accuracy_list[roi_index].append(1)
                else:
                    roi_accuracy_list[roi_index].append(0)
                #print '1,',
            else:
                if list_data_holder[row_index][1]==1:
                    roi_accuracy_list[roi_index].append(0)
                else:
                    roi_accuracy_list[roi_index].append(1)
                #print '0,',

        if sum_count>best_redundancy:
            guess_val=1
        else:
            guess_val=0

        #if guess_val==list_data_holder[row_index][1]:
        #    print '1'
        #else:
        #    print '0'

        if list_data_holder[row_index][1]==0:
            true_blank_guess_list.append(guess_val)
        elif list_data_holder[row_index][1]==1:
            true_print_guess_list.append(guess_val)
        else:
            print "************MARK ERROR********"
        if print_failures:
            if not guess_val==list_data_holder[row_index][1]:
                print dataframe_input.iloc[[row_index]]
    sensitivity=np.mean(true_print_guess_list)
    specificity=1-np.mean(true_blank_guess_list)
    J=sensitivity+specificity-1
    n_blank=len(true_blank_guess_list)
    n_print=len(true_print_guess_list)

    for lst in roi_accuracy_list:
        print np.average(lst),
        print ",",

    return [J, sensitivity, specificity, n_print, n_blank]

def cg_dataframe_to_list(dataframe_input,scan_size):
    header_list=list(dataframe_input)
    #scan_size=10
    roi_starter_list=["roi_0_scan_0","roi_1_scan_0","roi_2_scan_0","mark"]
    roi_starter_index_list=[0,0,0,0]
    for header_index in range(len(header_list)):
        for header_string_index in range(len(roi_starter_list)):
            if header_list[header_index]==roi_starter_list[header_string_index]:
                roi_starter_index_list[header_string_index]=header_index+1

    #print roi_starter_index_list
    
    #move values from dataframe to list: [[[roi1][roi2][roi3]],[mark]]
    list_data_holder=[]
    for row in dataframe_input.itertuples():
        #print row
        roi_list_holder=[]
        for roi_index in range(len(roi_starter_list)-1):
            roi_indi_holder=[]
            for scan_index in range(scan_size):
                roi_indi_holder.append(row[roi_starter_index_list[roi_index]+scan_index])
            roi_list_holder.append(roi_indi_holder)
        row_tuple=[roi_list_holder,row[roi_starter_index_list[3]]]
        list_data_holder.append(row_tuple)

    return list_data_holder
    

def cg_redundancy_modeler(dataframe_input,scan_size=10):
    scan_range=cg_scan_range_finder(dataframe_input,scan_size,3)
    #gonna sweep over bloody everything, and figure out the J value in each case, then save cases where J value is real good
    scan_n=20;
    roi0_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi1_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi2_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi0_dec_rng=np.linspace(-0.05,0.95,11)
    roi1_dec_rng=np.linspace(-0.05,0.95,11)
    roi2_dec_rng=np.linspace(-0.05,0.95,11) 
    redundancy_range=[0.5,1.5,2.5]
    roi_thresh_rng_list=[roi0_thresh_rng,roi1_thresh_rng,roi2_thresh_rng]
    roi_dec_rng_list=[roi0_dec_rng,roi1_dec_rng,roi2_dec_rng]
    

    header_list=list(dataframe_input)
    roi_starter_list=["roi_0_scan_0","roi_1_scan_0","roi_2_scan_0","mark"]
    roi_starter_index_list=[0,0,0,0]
    for header_index in range(len(header_list)):
        for header_string_index in range(len(roi_starter_list)):
            if header_list[header_index]==roi_starter_list[header_string_index]:
                roi_starter_index_list[header_string_index]=header_index+1

    #print roi_starter_index_list
    
    #move values from dataframe to list: [[[[roi1][roi2][roi3]]],[mark]]
    list_data_holder=[]
    for row in dataframe_input.itertuples():
        roi_list_holder=[]
        for roi_index in range(len(roi_thresh_rng_list)):
            roi_indi_holder=[]
            for scan_index in range(scan_size):
                roi_indi_holder.append(row[roi_starter_index_list[roi_index]+scan_index])
            roi_list_holder.append(roi_indi_holder)
        row_tuple=[roi_list_holder,row[roi_starter_index_list[3]]]
        list_data_holder.append(row_tuple)

    #print list_data_holder
    #first find optimal thresholds for each roi
    best_roi_thresh=len(roi_thresh_rng_list)*[-1]
    best_dec_thresh=len(roi_thresh_rng_list)*[-1]
    best_J=len(roi_thresh_rng_list)*[-1]
    for roi_index in range(len(roi_thresh_rng_list)):
        for roi_thresh in roi_thresh_rng_list[roi_index]:
            #make two new lists - blank confidence and print confidence
            print_confidence=[]
            blank_confidence=[]
            for row_index in range(len(list_data_holder)):
                #print row
                #print type(row)
                read_count=0
                for scan_index in range(len(list_data_holder[row_index][0][roi_index])):
                    #col_caller='roi_'+str(roi_index)+'_scan_'+str(scan_index)
                    
                    if list_data_holder[row_index][0][roi_index][scan_index]>roi_thresh:
                        read_count+=1
                confidence_value=float(read_count)/float(scan_size)
                row_marker=list_data_holder[row_index][1]
                #print row_marker
                if row_marker==1:
                    print_confidence.append(confidence_value)
                elif row_marker==0:
                    blank_confidence.append(confidence_value)
                else:
                    print "***********MARKER MISSING ERROR*********"
            ## now that the confidence lists are done, we get roi accuracy using dec_rng
            
            for roi_dec in roi_dec_rng_list[roi_index]:
                print_binary_list=np.zeros(len(print_confidence))
                blank_binary_list=np.zeros(len(blank_confidence))

                #set to 1 if value is greater than threshold
                print_binary_list[print_confidence>roi_dec]=1
                blank_binary_list[blank_confidence>roi_dec]=1



                sensitivity=np.mean(print_binary_list)
                specificity=1-np.mean(blank_binary_list)
                J=sensitivity+specificity-1

                if J>best_J[roi_index]:
                    best_J[roi_index]=J
                    best_roi_thresh[roi_index]=roi_thresh
                    best_dec_thresh[roi_index]=roi_dec
                    #print roi_index,
                    #print ",",
                    #print best_J[roi_index],
                    #print ",",
                    #print sensitivity,
                    #print ",",
                    #print specificity,
                    #print ",",
                    #print best_roi_thresh[roi_index],
                    #print ",",
                    #print best_dec_thresh[roi_index]
    #print best_J
    #print best_dec_thresh
    #print best_roi_thresh

    #now figure out the best level of redundancy
    print_sum=[]
    blank_sum=[]
    for row in dataframe_input.itertuples():
        sum=0
        for roi_index in range(len(roi_thresh_rng_list)):
            roi_thresh=best_roi_thresh[roi_index]
            dec_thresh=best_dec_thresh[roi_index]
            read_count=0
            for scan_index in range(scan_size):
                #col_caller='roi_'+str(roi_index)+'_scan_'+str(scan_index)
                    
                if row[roi_starter_index_list[roi_index]+scan_index]>roi_thresh:
                    read_count+=1
            confidence_value=float(read_count)/float(scan_size)
            if confidence_value>dec_thresh:
                sum+=1
        row_marker=row[roi_starter_index_list[3]]
        #print row_marker
        if row_marker==1:
            print_sum.append(sum)
        elif row_marker==0:
            blank_sum.append(sum)
        else:
            print "***********MARKER MISSING ERROR*********"
    #print print_sum
    #print blank_sum
    ## now that the confidence lists are done, we get roi accuracy using dec_rng
    best_redundancy=-1
    best_J=-1
    best_sensitivity=-1
    best_specificity=-1
    for redundancy in redundancy_range:
        print_binary_list=np.zeros(len(print_sum))
        blank_binary_list=np.zeros(len(blank_sum))

        #print redundancy
        #print print_sum
        #print print_binary_list
        #print blank_binary_list
        #set to 1 if value is greater than threshold
        for print_index in range(len(print_sum)):
            if print_sum[print_index]>redundancy:
                print_binary_list[print_index]=1

        for blank_index in range(len(blank_sum)):
            if blank_sum[blank_index]>redundancy:
                blank_binary_list[blank_index]=1

        #    print_binary_list[print_sum>redundancy]=1
        #blank_binary_list[blank_sum>redundancy]=1

        #print print_binary_list
        #print blank_binary_list

        sensitivity=np.mean(print_binary_list)
        specificity=1-np.mean(blank_binary_list)
        J=sensitivity+specificity-1

        #print redundancy,
        #print ",",
        #print J,
        #print ",",
        #print sensitivity,
        #print ",",
        #print specificity

        if J>best_J:
            best_J=J
            best_redundancy=redundancy
            best_sensitivity=sensitivity
            best_specificity=specificity


    #print "*********END*********"
    return [[best_J,best_sensitivity,best_specificity,len(print_sum),len(blank_sum)],best_roi_thresh,best_dec_thresh,best_redundancy]

def cg_redundancy_modeler_v2(dataframe_input,scan_size=10):
    scan_range=cg_scan_range_finder(dataframe_input,scan_size,3)
    #gonna sweep over bloody everything, and figure out the J value in each case, then save cases where J value is real good
    scan_n=20;
    #print scan_range[0]+3
    #print scan_range[1]-3
    roi0_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi1_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi2_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi0_dec_rng=np.linspace(-0.05,0.95,11)
    roi1_dec_rng=np.linspace(-0.05,0.95,11)
    roi2_dec_rng=np.linspace(-0.05,0.95,11) 
    redundancy_range=[0.5,1.5,2.5]
    roi_thresh_rng_list=[roi0_thresh_rng,roi1_thresh_rng,roi2_thresh_rng]
    roi_dec_rng_list=[roi0_dec_rng,roi1_dec_rng,roi2_dec_rng]

    header_list=list(dataframe_input)
    roi_starter_list=["roi_0_scan_0","roi_1_scan_0","roi_2_scan_0","mark"]
    roi_starter_index_list=[0,0,0,0]
    for header_index in range(len(header_list)):
        for header_string_index in range(len(roi_starter_list)):
            if header_list[header_index]==roi_starter_list[header_string_index]:
                roi_starter_index_list[header_string_index]=header_index+1

    #print roi_starter_index_list
    
    #move values from dataframe to list: [[[[roi1][roi2][roi3]]],[mark]]
    list_data_holder=[]
    for row in dataframe_input.itertuples():
        roi_list_holder=[]
        for roi_index in range(len(roi_thresh_rng_list)):
            roi_indi_holder=[]
            for scan_index in range(scan_size):
                roi_indi_holder.append(row[roi_starter_index_list[roi_index]+scan_index])
            roi_list_holder.append(roi_indi_holder)
        row_tuple=[roi_list_holder,row[roi_starter_index_list[3]]]
        list_data_holder.append(row_tuple)


    #print list_data_holder
    #first find optimal thresholds for each roi
    best_roi_thresh=len(roi_thresh_rng_list)*[-1]
    best_dec_thresh=len(roi_thresh_rng_list)*[-1]
    best_J=len(roi_thresh_rng_list)*[-1]
    for roi_index in range(len(roi_thresh_rng_list)):
        for roi_thresh in roi_thresh_rng_list[roi_index]:
            #make two new lists - blank confidence and print confidence
            print_confidence=[]
            blank_confidence=[]
            for row_index in range(len(list_data_holder)):
                #print row
                #print type(row)
                read_count=0
                for scan_index in range(len(list_data_holder[row_index][0][roi_index])):
                    #col_caller='roi_'+str(roi_index)+'_scan_'+str(scan_index)
                    
                    if list_data_holder[row_index][0][roi_index][scan_index]>roi_thresh:
                        read_count+=1
                confidence_value=float(read_count)/float(scan_size)
                row_marker=list_data_holder[row_index][1]
                #print row_marker
                if row_marker==1:
                    print_confidence.append(confidence_value)
                elif row_marker==0:
                    blank_confidence.append(confidence_value)
                else:
                    print "***********MARKER MISSING ERROR*********"
            ## now that the confidence lists are done, we get roi accuracy using dec_rng
            
            for roi_dec in roi_dec_rng_list[roi_index]:
                print_binary_list=np.zeros(len(print_confidence))
                blank_binary_list=np.zeros(len(blank_confidence))

                #set to 1 if value is greater than threshold
                print_binary_list[print_confidence>roi_dec]=1
                blank_binary_list[blank_confidence>roi_dec]=1



                sensitivity=np.mean(print_binary_list)
                specificity=1-np.mean(blank_binary_list)
                J=adapted_J(sensitivity,specificity)
                #J=sensitivity+specificity-1

                if J>best_J[roi_index]:
                    T_list=[]
                    D_list=[]
                    best_J[roi_index]=J
                    T_list.append(roi_thresh)
                    D_list.append(roi_dec)
                    #best_roi_thresh[roi_index]=roi_thresh
                    #best_dec_thresh[roi_index]=roi_dec
                elif J==best_J[roi_index]:
                    T_list.append(roi_thresh)
                    D_list.append(roi_dec)
        #once all threshold-dec pairs have been tried, take median of each of T_list and D_list vectors
        best_roi_thresh[roi_index]=np.median(T_list)
        best_dec_thresh[roi_index]=np.median(D_list)
        #print len(T_list)
        #print len(D_list)

    #print [best_J,best_roi_thresh,best_dec_thresh]
    scan_n=100
    T_swing=0.5
    D_swing=0.4
    roi0_thresh_rng=np.linspace(best_roi_thresh[0]-T_swing,best_roi_thresh[0]+T_swing,scan_n)
    roi1_thresh_rng=np.linspace(best_roi_thresh[1]-T_swing,best_roi_thresh[1]+T_swing,scan_n)
    roi2_thresh_rng=np.linspace(best_roi_thresh[2]-T_swing,best_roi_thresh[2]+T_swing,scan_n)
    roi0_dec_rng=np.arange(np.max([-0.05,best_dec_thresh[0]-D_swing]),np.min([0.95,best_dec_thresh[0]+D_swing])+0.1,0.1)
    roi1_dec_rng=np.arange(np.max([-0.05,best_dec_thresh[1]-D_swing]),np.min([0.95,best_dec_thresh[1]+D_swing])+0.1,0.1)
    roi2_dec_rng=np.arange(np.max([-0.05,best_dec_thresh[2]-D_swing]),np.min([0.95,best_dec_thresh[2]+D_swing])+0.1,0.1)
    #print roi0_dec_rng
    roi_thresh_rng_list=[roi0_thresh_rng,roi1_thresh_rng,roi2_thresh_rng]
    roi_dec_rng_list=[roi0_dec_rng,roi1_dec_rng,roi2_dec_rng]
        #print list_data_holder
    #first find optimal thresholds for each roi
    best_roi_thresh=len(roi_thresh_rng_list)*[-1]
    best_dec_thresh=len(roi_thresh_rng_list)*[-1]
    #best_J=len(roi_thresh_rng_list)*[-1]
    for roi_index in range(len(roi_thresh_rng_list)):
        T_list=[]
        D_list=[]
        for roi_thresh in roi_thresh_rng_list[roi_index]:
            #make two new lists - blank confidence and print confidence
            print_confidence=[]
            blank_confidence=[]
            for row_index in range(len(list_data_holder)):
                #print row
                #print type(row)
                read_count=0
                for scan_index in range(len(list_data_holder[row_index][0][roi_index])):
                    #col_caller='roi_'+str(roi_index)+'_scan_'+str(scan_index)
                    
                    if list_data_holder[row_index][0][roi_index][scan_index]>roi_thresh:
                        read_count+=1
                confidence_value=float(read_count)/float(scan_size)
                row_marker=list_data_holder[row_index][1]
                #print row_marker
                if row_marker==1:
                    print_confidence.append(confidence_value)
                elif row_marker==0:
                    blank_confidence.append(confidence_value)
                else:
                    print "***********MARKER MISSING ERROR*********"
            ## now that the confidence lists are done, we get roi accuracy using dec_rng
            
            for roi_dec in roi_dec_rng_list[roi_index]:
                print_binary_list=np.zeros(len(print_confidence))
                blank_binary_list=np.zeros(len(blank_confidence))

                #set to 1 if value is greater than threshold
                print_binary_list[print_confidence>roi_dec]=1
                blank_binary_list[blank_confidence>roi_dec]=1



                sensitivity=np.mean(print_binary_list)
                specificity=1-np.mean(blank_binary_list)
                J=sensitivity+specificity-1
                J=adapted_J(sensitivity,specificity)

                if J>best_J[roi_index]-0.05:
                    T_list.append(roi_thresh)
                    D_list.append(roi_dec)
        #once all threshold-dec pairs have been tried, take median of each of T_list and D_list vectors
        best_roi_thresh[roi_index]=np.median(T_list)
        best_dec_thresh[roi_index]=np.median(D_list)
        #print len(T_list)
        #print len(D_list)


                    #print roi_index,
                    #print ",",
                    #print best_J[roi_index],
                    #print ",",
                    #print sensitivity,
                    #print ",",
                    #print specificity,
                    #print ",",
                    #print roi_thresh,
                    #print ",",
                    #print roi_dec
    #print best_J
    #print best_dec_thresh
    #print best_roi_thresh

    #now figure out the best level of redundancy
    print_sum=[]
    blank_sum=[]
    for row in dataframe_input.itertuples():
        sum=0
        for roi_index in range(len(roi_thresh_rng_list)):
            roi_thresh=best_roi_thresh[roi_index]
            dec_thresh=best_dec_thresh[roi_index]
            read_count=0
            for scan_index in range(scan_size):
                #col_caller='roi_'+str(roi_index)+'_scan_'+str(scan_index)
                    
                if row[roi_starter_index_list[roi_index]+scan_index]>roi_thresh:
                    read_count+=1
            confidence_value=float(read_count)/float(scan_size)
            if confidence_value>dec_thresh:
                sum+=1
        row_marker=row[roi_starter_index_list[3]]
        #print row_marker
        if row_marker==1:
            print_sum.append(sum)
        elif row_marker==0:
            blank_sum.append(sum)
        else:
            print "***********MARKER MISSING ERROR*********"
    #print print_sum
    #print blank_sum
    ## now that the confidence lists are done, we get roi accuracy using dec_rng
    best_redundancy=-1
    best_J=-1
    best_sensitivity=-1
    best_specificity=-1
    for redundancy in redundancy_range:
        print_binary_list=np.zeros(len(print_sum))
        blank_binary_list=np.zeros(len(blank_sum))

        #print redundancy
        #print print_sum
        #print print_binary_list
        #print blank_binary_list
        #set to 1 if value is greater than threshold
        for print_index in range(len(print_sum)):
            if print_sum[print_index]>redundancy:
                print_binary_list[print_index]=1

        for blank_index in range(len(blank_sum)):
            if blank_sum[blank_index]>redundancy:
                blank_binary_list[blank_index]=1

        #    print_binary_list[print_sum>redundancy]=1
        #blank_binary_list[blank_sum>redundancy]=1

        #print print_binary_list
        #print blank_binary_list

        sensitivity=np.mean(print_binary_list)
        specificity=1-np.mean(blank_binary_list)
        J=sensitivity+specificity-1

        #print redundancy,
        #print ",",
        #print J,
        #print ",",
        #print sensitivity,
        #print ",",
        #print specificity

        if J>best_J:
            best_J=J
            best_redundancy=redundancy
            best_sensitivity=sensitivity
            best_specificity=specificity


    #print "*********END*********"
    return [[best_J,best_sensitivity,best_specificity,len(print_sum),len(blank_sum)],best_roi_thresh,best_dec_thresh,best_redundancy]

def cg_redundancy_modeler_v3(dataframe_input,scan_size=10,roi_total=3):
    scan_range=cg_scan_range_finder(dataframe_input,scan_size,3)
    #reset index of input dataframe
    dataframe_input=dataframe_input.reset_index(drop=True)


    #gonna sweep over bloody everything, and figure out the J value in each case, then save cases where J value is real good
    scan_n=100;
    #print scan_range[0]+3
    #print scan_range[1]-3
    roi0_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi1_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi2_thresh_rng=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi0_dec_rng=np.linspace(-0.05,0.95,11)
    roi1_dec_rng=np.linspace(-0.05,0.95,11)
    roi2_dec_rng=np.linspace(-0.05,0.95,11) 
    redundancy_range=np.linspace(0.5,roi_total-0.5,roi_total)
    roi_thresh_rng_list=[roi0_thresh_rng,roi1_thresh_rng,roi2_thresh_rng]
    roi_dec_rng_list=[roi0_dec_rng,roi1_dec_rng,roi2_dec_rng]

    #New code that can accomidate n number of rois
    roi_thresh_rng_list=[]
    roi_thresh_basis=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi_dec_rng_list=[]
    roi_dec_basis=np.linspace(-0.05,0.95,scan_size+1)
   # print roi_dec_basis

    for roi_iterator in range(roi_total):
        roi_thresh_rng_list.append(roi_thresh_basis)
        roi_dec_rng_list.append(roi_dec_basis)




    header_list=list(dataframe_input)
    roi_starter_list=["roi_0_scan_0","roi_1_scan_0","roi_2_scan_0","mark"]
    roi_starter_list=[]
    roi_starter_index_list=[]
    for roi_iterator in range(roi_total):
        roi_starter_list.append("roi_"+str(roi_iterator)+"_scan_0")
        roi_starter_index_list.append(0)
    roi_starter_list.append("mark")
    roi_starter_index_list.append(0)
    #roi_starter_index_list=[0,0,0,0]
    for header_index in range(len(header_list)):
        for header_string_index in range(len(roi_starter_list)):
            if header_list[header_index]==roi_starter_list[header_string_index]:
                roi_starter_index_list[header_string_index]=header_index+1

    #print roi_starter_index_list
    
   

    #move values from dataframe to list: [[[[roi1][roi2][roi3]]],[mark]]
    list_data_holder=[]
    for row in dataframe_input.itertuples():
        roi_list_holder=[]
        for roi_index in range(len(roi_thresh_rng_list)):
            roi_indi_holder=[]
            for scan_index in range(scan_size):
                roi_indi_holder.append(row[roi_starter_index_list[roi_index]+scan_index])
            roi_list_holder.append(roi_indi_holder)
        row_tuple=[roi_list_holder,row[roi_starter_index_list[-1]]]
        list_data_holder.append(row_tuple)


    #print list_data_holder

    #iter_temp=0
    #for row in list_data_holder:
    #    print row[1],
    #    print "  "+str(iter_temp)
    #    iter_temp+=1


    #first find optimal thresholds for each roi
    best_roi_thresh=len(roi_thresh_rng_list)*[-1]
    best_dec_thresh=len(roi_thresh_rng_list)*[-1]
    best_J=len(roi_thresh_rng_list)*[-1]
    for roi_index in range(len(roi_thresh_rng_list)):
        for roi_thresh in roi_thresh_rng_list[roi_index]:
            #make two new lists - blank confidence and print confidence
            print_confidence=[]
            blank_confidence=[]
            for row_index in range(len(list_data_holder)):
                #print row
                #print type(row)
                read_count=0
                for scan_index in range(len(list_data_holder[row_index][0][roi_index])):
                    #col_caller='roi_'+str(roi_index)+'_scan_'+str(scan_index)
                    
                    if list_data_holder[row_index][0][roi_index][scan_index]>roi_thresh:
                        read_count+=1
                confidence_value=float(read_count)/float(scan_size)
                row_marker=list_data_holder[row_index][1]
                #print row_marker
                if row_marker==1:
                    print_confidence.append(confidence_value)
                elif row_marker==0:
                    blank_confidence.append(confidence_value)
                else:
                    print "***********MARKER MISSING ERROR1*********"
            ## now that the confidence lists are done, we get roi accuracy using dec_rng
            
            for roi_dec in roi_dec_rng_list[roi_index]:
                print_binary_list=np.zeros(len(print_confidence))
                blank_binary_list=np.zeros(len(blank_confidence))

                #set to 1 if value is greater than threshold
                print_binary_list[print_confidence>roi_dec]=1
                blank_binary_list[blank_confidence>roi_dec]=1



                sensitivity=np.mean(print_binary_list)
                specificity=1-np.mean(blank_binary_list)
                J=adapted_J(sensitivity,specificity)
                #J=sensitivity+specificity-1

                if J>best_J[roi_index]:
                    T_list=[]
                    D_list=[]
                    best_J[roi_index]=J
                    T_list.append(roi_thresh)
                    D_list.append(roi_dec)
                    #best_roi_thresh[roi_index]=roi_thresh
                    #best_dec_thresh[roi_index]=roi_dec
                elif J==best_J[roi_index]:
                    T_list.append(roi_thresh)
                    D_list.append(roi_dec)
        #once all threshold-dec pairs have been tried, take median of each of T_list and D_list vectors
        best_roi_thresh[roi_index]=np.median(T_list)
        best_dec_thresh[roi_index]=np.median(D_list)
        #print len(T_list)
        #print len(D_list)

    #print [best_J,best_roi_thresh,best_dec_thresh]
    scan_n=100
    T_swing=0.5
    D_swing=0.4
    #roi0_thresh_rng=np.linspace(best_roi_thresh[0]-T_swing,best_roi_thresh[0]+T_swing,scan_n)
    #roi1_thresh_rng=np.linspace(best_roi_thresh[1]-T_swing,best_roi_thresh[1]+T_swing,scan_n)
    #roi2_thresh_rng=np.linspace(best_roi_thresh[2]-T_swing,best_roi_thresh[2]+T_swing,scan_n)
    #roi0_dec_rng=np.arange(np.max([-0.05,best_dec_thresh[0]-D_swing]),np.min([0.95,best_dec_thresh[0]+D_swing])+0.1,0.1)
    #roi1_dec_rng=np.arange(np.max([-0.05,best_dec_thresh[1]-D_swing]),np.min([0.95,best_dec_thresh[1]+D_swing])+0.1,0.1)
    #roi2_dec_rng=np.arange(np.max([-0.05,best_dec_thresh[2]-D_swing]),np.min([0.95,best_dec_thresh[2]+D_swing])+0.1,0.1)
    ##print roi0_dec_rng
    #roi_thresh_rng_list=[roi0_thresh_rng,roi1_thresh_rng,roi2_thresh_rng]
    #roi_dec_rng_list=[roi0_dec_rng,roi1_dec_rng,roi2_dec_rng]

    #New code that can accomidate n number of rois
    roi_thresh_rng_list=[]
    roi_thresh_basis=np.linspace(scan_range[0],scan_range[1],scan_n)
    roi_dec_rng_list=[]
    roi_dec_basis=np.linspace(-0.05,0.95,scan_size+1)
   # print roi_dec_basis

    for roi_iterator in range(roi_total):
        thresh_list_temp=np.linspace(best_roi_thresh[roi_iterator]-T_swing,best_roi_thresh[roi_iterator]+T_swing,scan_n)
        roi_thresh_rng_list.append(thresh_list_temp)
        dec_list_temp=np.arange(np.max([-0.05,best_dec_thresh[roi_iterator]-D_swing]),np.min([0.95,best_dec_thresh[roi_iterator]+D_swing])+0.1,0.1)
       # roi_dec_rng_list.append(dec_list_temp)
        roi_dec_rng_list.append(roi_dec_basis)

  #  print roi_dec_rng_list


        #print list_data_holder
    #first find optimal thresholds for each roi
    best_roi_thresh=len(roi_thresh_rng_list)*[-1]
    best_dec_thresh=len(roi_thresh_rng_list)*[-1]
    #best_J=len(roi_thresh_rng_list)*[-1]
    for roi_index in range(len(roi_thresh_rng_list)):
        T_list=[]
        D_list=[]
        for roi_thresh in roi_thresh_rng_list[roi_index]:
            #make two new lists - blank confidence and print confidence
            print_confidence=[]
            blank_confidence=[]
            for row_index in range(len(list_data_holder)):
                #print row
                #print type(row)
                read_count=0
                for scan_index in range(len(list_data_holder[row_index][0][roi_index])):
                    #col_caller='roi_'+str(roi_index)+'_scan_'+str(scan_index)
                    
                    if list_data_holder[row_index][0][roi_index][scan_index]>roi_thresh:
                        read_count+=1
                confidence_value=float(read_count)/float(scan_size)
                row_marker=list_data_holder[row_index][1]
                #print row_marker
                if row_marker==1:
                    print_confidence.append(confidence_value)
                elif row_marker==0:
                    blank_confidence.append(confidence_value)
                else:
                    print "***********MARKER MISSING ERROR2*********"
            ## now that the confidence lists are done, we get roi accuracy using dec_rng
            
            for roi_dec in roi_dec_rng_list[roi_index]:
                print_binary_list=np.zeros(len(print_confidence))
                blank_binary_list=np.zeros(len(blank_confidence))

                #set to 1 if value is greater than threshold
                print_binary_list[print_confidence>roi_dec]=1
                blank_binary_list[blank_confidence>roi_dec]=1



                sensitivity=np.mean(print_binary_list)
                specificity=1-np.mean(blank_binary_list)
                J=sensitivity+specificity-1
                J=adapted_J(sensitivity,specificity)

                if J>best_J[roi_index]-0.05:
                    T_list.append(roi_thresh)
                    D_list.append(roi_dec)
        #once all threshold-dec pairs have been tried, take median of each of T_list and D_list vectors
        best_roi_thresh[roi_index]=np.median(T_list)
        best_dec_thresh[roi_index]=np.median(D_list)
        #print len(T_list)
        #print len(D_list)


                    #print roi_index,
                    #print ",",
                    #print best_J[roi_index],
                    #print ",",
                    #print sensitivity,
                    #print ",",
                    #print specificity,
                    #print ",",
                    #print roi_thresh,
                    #print ",",
                    #print roi_dec
    print best_J,
    print ";",
    #print best_dec_thresh
    #print best_roi_thresh

        #print print_sum
    #print blank_sum
    ## now that the confidence lists are done, we get roi accuracy using dec_rng
    accuracy_list=[]
    best_redundancy=-1
    best_J=-1
    best_sensitivity=-1
    best_specificity=-1
    active_rois_final=[]
    available_ROIs=range(roi_total)

    for ROI_count in range(1,roi_total+1):
        #ROI_count=1
        for active_ROIs in list(it.combinations(range(0,len(available_ROIs)),ROI_count)):

            #now figure out the best level of redundancy
            print_sum=[]
            blank_sum=[]
            sum_list=[]
            mark_list=[]
            for row in dataframe_input.itertuples():
                sum=0
                for roi_index in active_ROIs:
                    roi_thresh=best_roi_thresh[roi_index]
                    dec_thresh=best_dec_thresh[roi_index]
                    read_count=0
                    for scan_index in range(scan_size):
                        #col_caller='roi_'+str(roi_index)+'_scan_'+str(scan_index)
                    
                        if row[roi_starter_index_list[roi_index]+scan_index]>roi_thresh:
                            read_count+=1
                    confidence_value=float(read_count)/float(scan_size)
                    if confidence_value>dec_thresh:
                        sum+=1
                row_marker=row[roi_starter_index_list[-1]]
                mark_list.append(row_marker)
                #print row_marker
                if row_marker==1:
                    print_sum.append(sum)
                    sum_list.append(sum)
            
                elif row_marker==0:
                    blank_sum.append(sum)
                    sum_list.append(sum)
                else:
                    print "***********MARKER MISSING ERROR3*********"

            for redundancy in redundancy_range:
                print_binary_list=np.zeros(len(print_sum))
                blank_binary_list=np.zeros(len(blank_sum))

                #print redundancy
                #print print_sum
                #print print_binary_list
                #print blank_binary_list
                #set to 1 if value is greater than threshold
                for print_index in range(len(print_sum)):
                    if print_sum[print_index]>redundancy:
                        print_binary_list[print_index]=1

                for blank_index in range(len(blank_sum)):
                    if blank_sum[blank_index]>redundancy:
                        blank_binary_list[blank_index]=1

                #    print_binary_list[print_sum>redundancy]=1
                #blank_binary_list[blank_sum>redundancy]=1

                #print print_binary_list
                #print blank_binary_list

                sensitivity=np.mean(print_binary_list)
                specificity=1-np.mean(blank_binary_list)
                J=sensitivity+specificity-1

                #print redundancy,
                #print ",",
                #print J,
                #print ",",
                #print sensitivity,
                #print ",",
                #print specificity

                if J>best_J:
                    best_J=J
                    best_redundancy=redundancy
                    best_sensitivity=sensitivity
                    best_specificity=specificity
                    active_rois_final=active_ROIs


    save_failed_images(dataframe_input,mark_list,sum_list,best_redundancy)

    #print "*********END*********"
    return [[best_J,best_sensitivity,best_specificity,len(print_sum),len(blank_sum)],best_roi_thresh,best_dec_thresh,best_redundancy,active_rois_final]

def logistic_regression_prep_cg(dataframe_input,dataframe_blank,roi,roi_max,combine_scan_data=True):
    df=copy.copy(dataframe_input)

    #PULL OUT THE DATA YOU WANT
    print_list=[]
    blank_list=[]
    for index in range(roi_max):
        col_caller='roi_'+str(roi)+'_scan_'+str(index)
        print_list.extend(df[col_caller].tolist())
        blank_list.extend(dataframe_blank[col_caller].tolist())

    print_y_val=len(print_list)*[1]
    blank_y_val=len(blank_list)*[0]

    x_col_list=[]
    x_col_list.extend(blank_list)
    x_col_list.extend(print_list)

    y_col_list=[]
    y_col_list.extend(print_y_val)
    y_col_list.extend(blank_y_val)

    y_mean=np.mean(print_list)
    x_mean=np.mean(blank_list)
    print x_mean,
    print ",",
    #print len(x_col_list)
    #print len(y_col_list)

    #x_col_df=pd.DataFrame({'roi_'+str(roi):x_col_list})
    x_col_df=pd.DataFrame({'roi':x_col_list})
    y_col_df=pd.DataFrame({'mark':y_col_list})


    
    return [x_col_df,y_col_df]


def logistic_regression_model(X_train,y_train, printer=False,tester_switch=False,xTst=[],yTst=[],overfitting_analysis=False,confidence_table=False):
   
    classifier = LogisticRegression(solver='newton-cg', random_state = 0,fit_intercept=True,class_weight="balanced")

    classifier.fit(X_train, y_train)

    if tester_switch:
        X_train=xTst
        y_train=yTst

    y_pred = classifier.predict(X_train)
    if overfitting_analysis:
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import recall_score,f1_score,log_loss,roc_auc_score
        from sklearn.metrics import mean_squared_error
        scoring=['recall','neg_mean_squared_error','f1','neg_log_loss','roc_auc']
        scores=cross_validate(classifier,X_train,y_train,scoring=scoring,cv=10,return_train_score=False)
        print np.average(scores['test_recall']),
        print ",",
        print np.average(scores['test_neg_mean_squared_error']),
        print ",",
        print np.average(scores['test_f1']),
        print ",",
        print np.average(scores['test_neg_log_loss']),
        print ",",
        print np.average(scores['test_roc_auc']),
    


        from sklearn.metrics import classification_report
        report=classification_report(y_train,y_pred,digits=4)
        print report

    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_train, y_pred) 
    #print confusion_matrix
    tn=confusion_matrix[0,0]
    fp=confusion_matrix[0,1]
    tp=confusion_matrix[1,1]
    fn=confusion_matrix[1,0]
    tpr=float(tp)/float(tp+fn)
    tnr=float(tn)/float(tn+fp)
    sensitivity=float(tp)/float(tp+fn)
    specificity=float(tn)/float(tn+fp)
    fpr=fp/float(fp+tn)
    #truth=tpr-(1-tnr)
    J=sensitivity+specificity-1
    truth=J
    
    #if printer:
    #    print 'Truth: ' + str(truth)
    #    print 'TPP: '+str(tpr)
    #    print 'fpr: '+str(fpr)

    if confidence_table:
        for row in range(0,len(X_train.index)):
            mark_perc= classifier.predict_proba(X_train.iloc[[row]])[0,1]
            LUX= X_train.iloc[[row]].values[0,14]
            print " ",
            mark= y_train.iloc[[row]].values[0]
            if mark_perc<0.5:
                pred_mark=0
                confidence=(0.5-mark_perc)*2
            else:
                pred_mark=1
                confidence=(mark_perc-0.5)*2
            if pred_mark==mark:
                accuracy=1
            else:
                accuracy=0
            multiple1=100
            multiple2=300
            if LUX<=1000:
                MultLux=np.floor((LUX + multiple1/2) / multiple1) * multiple1
            else:
                MultLux=np.floor((LUX + multiple2/2) / multiple2) * multiple2
            MultLux=int(MultLux)
            print str(mark_perc)+","+str(LUX)+","+str(MultLux)+","+str(mark)+","+str(pred_mark)+","+str(accuracy)+","+str(confidence)

        print(confusion_matrix)
        print y_train.mean()
        print classifier.coef_
        print classifier.intercept_

    if printer:
        print(str(data.shape[0])+', {:.5f}'.format(classifier.score(X_train, y_train)))+","+str(truth)+ "," + str(tpr) + "," + str(fpr) + "," + str(tp) + "," + str(fn) + "," + str(tn) + "," + str(fp)
    return [J,sensitivity,specificity,classifier.coef_,classifier.intercept_]

def sm_logistic_regression_model(X_train,y_train, printer=False,tester_switch=False,xTst=[],yTst=[]):
   
    X_train=sm_tool.add_constant(X_train)

    classifier = sm.Logit(y_train,X_train)

    result=classifier.fit(disp=0,warn_convergence=False)

    coeffs=result.params.values

    #print coeffs
    if printer:
        print "space"
        print result.summary()

        print result.pred_table(0.5)
    
    
    
    prediction_train=result.predict(X_train)
    
    pred_train=sm_logistic_model_tester(prediction_train,y_train)
    #print pred_train
    if not len(xTst)==0:
        xTst=sm_tool.add_constant(xTst)
        prediction_test=result.predict(xTst)
    
        pred_test=sm_logistic_model_tester(prediction_test,yTst)
        if printer:

            print len(prediction_train)
            print len(prediction_test)

   
        #print pred_test
        print min([pred_train[0],pred_test[0]]),
        print "]",
        print pred_train[0],
        print "]",
        print pred_test[0],
        print "]",
        print pred_test[1],
        print "]",
        print pred_test[2],
        print "]",
        print pred_train[1],
        print "]",
        print pred_train[2]
        #print prediction

        return [pred_train[0],pred_test[0],pred_test[1],pred_test[2],pred_train[1],pred_train[2]]
    else:
        return[pred_train[0],pred_train[1],pred_train[2]]

def sm_logistic_model_tester(mark_perc_vec,y_train):
    y_train_list=y_train.values.tolist()
    #print y_train_list
    if not len(mark_perc_vec)==len(y_train_list):
        print "**********ERROR**********"
        print len(mark_perc_vec)
        print len(y_train_list)
    tp_counter=0;
    fp_counter=0;
    tn_counter=0;
    fn_counter=0;
    accuracy_holder=[]
    for index in range(len(mark_perc_vec)):
        if y_train_list[index]==0:
            if mark_perc_vec[index]>0.5:
                fp_counter+=1
                accuracy_holder.append(0)
            elif mark_perc_vec[index]<=0.5:
                tn_counter+=1
                accuracy_holder.append(1)
        elif y_train_list[index]==1:
            if mark_perc_vec[index]>0.5:
                tp_counter+=1
                accuracy_holder.append(1)
            elif mark_perc_vec[index]<=0.5:
                fn_counter+=1
                accuracy_holder.append(0)
    Sen=tp_counter/float(tp_counter+fn_counter)
    Spec=tn_counter/float(fp_counter+tn_counter)
    J=Sen+Spec-1
    return J,Sen,Spec,tp_counter,tn_counter,fp_counter,fn_counter

def threshold_finder(input_data, input_mark,steps=100):   
    #print input_data
    #print input_mark
    if not len(input_data)==len(input_mark):
        raise ValueError('Input mark and data are not of equal size')
    sweep_start=np.percentile(input_data,10)
    sweep_end=np.percentile(input_data,90)


    #only sweep over data where blank and print sets overlap
    blank_print_range=[[],[]]
   
    for iter in range(len(input_data)):
        blank_print_switch= int(input_mark[iter])
        #print blank_print_switch
        #print blank_print_range[blank_print_switch]
        if not blank_print_range[blank_print_switch]: #range is empty
            temp_list=[input_data[iter],input_data[iter]]
            blank_print_range[blank_print_switch]=temp_list
            #blank_print_range[blank_print_switch][0]=input_data[iter]
            #blank_print_range[blank_print_switch][1]=input_data[iter]
        else:
            if input_data[iter]<blank_print_range[blank_print_switch][0]:
                blank_print_range[blank_print_switch][0]=input_data[iter]
            if input_data[iter]>blank_print_range[blank_print_switch][1]:
                blank_print_range[blank_print_switch][1]=input_data[iter]
    
    #print "BP Range: "
    #print blank_print_range

    if blank_print_range[0][1]<blank_print_range[1][0]: #NO OVERLAP
        steps=1
        middle_value=np.average([blank_print_range[0][1],blank_print_range[1][0]])
        sweep_start=middle_value
        sweep_end=middle_value+1 #doesn't matter what this is, but don't want it to equal the start value because that will mess up the min_step_check later
    elif blank_print_range[0][0]>blank_print_range[1][1]: #NO OVERLAP
        steps=1
        middle_value=np.average([blank_print_range[0][0],blank_print_range[1][1]])
        sweep_start=middle_value
        sweep_end=middle_value+1 #doesn't matter what this is, but don't want it to equal the start value because that will mess up the min_step_check later
    else:
        sweep_start=np.max([blank_print_range[0][0],blank_print_range[1][0]])
        sweep_end=np.min([blank_print_range[0][1],blank_print_range[1][1]])
    
    #check if step is too small
    min_step=0.05
    if (sweep_end-sweep_start)/float(steps)<min_step:
        steps=int((sweep_end-sweep_start)/float(min_step))

    #print sweep_start,
    #print ",",
    #print sweep_end,
    #print ",",
    #print steps

    sweep=np.linspace(sweep_start,sweep_end,steps)

    J_max=0
    thresh_best=-999
    sen_best=-999
    spec_best=-999
    for thresh in sweep:
        tp=0
        fp=0
        tn=0
        fn=0
        for data_index in range(len(input_data)):
            if input_data[data_index]<thresh:
                mark_guess=0
            else:
                mark_guess=1
            if input_mark[data_index] == 0:
                if mark_guess==0:
                    tn+=1
                else:
                    fp+=1
            else:
                if mark_guess==0:
                    fn+=1
                else:
                    tp+=1
        sen=float(tp)/float((tp+fn))
        spec=float(tn)/float((tn+fp))
        J=adapted_J(sen,spec)
        if abs(J)>abs(J_max):
            J_max=J
            thresh_best=thresh
            sen_best=sen
            spec_best=spec
            tp_best=tp
            fp_best=fp
            tn_best=tn
            fn_best=fn

    #print [thresh_best,abs(J_max),J_max,sen_best,spec_best,tp_best,fp_best,tn_best,fn_best]

    return [thresh_best,abs(J_max),J_max,sen_best,spec_best]

def threshold_tester(input_data, input_mark,thresh):   
    #print input_data
    #print input_mark
    if not len(input_data)==len(input_mark):
        raise ValueError('Input mark and data are not of equal size')

    accuracy_list=[]
    guess_holder=[]
    
    tp=0
    fp=0
    tn=0
    fn=0
    for data_index in range(len(input_data)):
        if input_data[data_index]<thresh:
            mark_guess=0
            guess_holder.append(0)
        else:
            mark_guess=1
            guess_holder.append(1)
        if input_mark[data_index] == 0:
            if mark_guess==0:
                tn+=1
                accuracy_list.append(1)
            else:
                fp+=1
                accuracy_list.append(0)
        else:
            if mark_guess==0:
                fn+=1
                accuracy_list.append(0)
            else:
                tp+=1
                accuracy_list.append(1)
    sen=float(tp)/float((tp+fn))
    spec=float(tn)/float((tn+fp))
    J=adapted_J(sen,spec)


    #print [thresh_best,abs(J_max),J_max,sen_best,spec_best,tp_best,fp_best,tn_best,fn_best]

    return J,guess_holder,accuracy_list

def logistic_model_tester(X_train,y_train,coeffs,intercept):
    x_train_list=X_train.values.tolist()
    y_train_list=y_train.values.tolist()
    logistic_result=[]
    for index in range(len(x_train_list)):
        
        log_res=logistic_percent_calculator(coeffs,intercept,x_train_list[index])
        #print x_train_list[index],
        #print log_res,
        #print ",",
        #print y_train_list[index]
        if log_res>0.5:
            mark_pred=1
        else:
            mark_pred=0
        logistic_result.append([y_train_list[index],mark_pred])
    tp_counter=0;
    fp_counter=0;
    tn_counter=0;
    fn_counter=0;
    accuracy_holder=[]
    for mark_pair in logistic_result:
        if mark_pair[0]==0:
            if mark_pair[1]==1:
                fp_counter+=1
                accuracy_holder.append(0)
            elif mark_pair[1]==0:
                tn_counter+=1
                accuracy_holder.append(1)
        elif mark_pair[0]==1:
            if mark_pair[1]==1:
                tp_counter+=1
                accuracy_holder.append(1)
            elif mark_pair[1]==0:
                fn_counter+=1
                accuracy_holder.append(0)

    tpr=tp_counter/float(tp_counter+fn_counter)
    fpr=fp_counter/float(fp_counter+tn_counter)
    truth=tpr-fpr
    return truth,tpr,fpr,tp_counter,tn_counter,fp_counter,fn_counter,accuracy_holder,logistic_result


def path_filter(dataframe_input,path):
    df=copy.copy(dataframe_input)

    if not path=='skip':
        df=df.loc[df['path'].str.contains(path)]

    return df

def arbitrary_exclude(dataframe_input,column,input_string):
    df=copy.copy(dataframe_input)

    

    if not input_string=='skip':
        df[column]=df[column].astype(str)
        df = df[~df[column].isin([input_string])]

    return df

def arbitrary_include(dataframe_input,column,input_string):
    df=copy.copy(dataframe_input)
    if not input_string=='skip':
        df[column]=df[column].astype(str)
        #print df[column]
        #print input_string
        if column=='path':
            df=df.loc[df[column].str.contains(input_string)]
            print 'path detected'
        else:
            df = df[df[column]==input_string]

    return df

def arbitrary_filter(dataframe_input,column,str):
    df=copy.copy(dataframe_input)

    if not str=='skip':
        df=df.loc[df[column].str.contains(str)]

    return df

def pk_modeler(dataframe_input,ring_count,test_type=[1]):
    #code will accept a dataframe input and ring count.  It'll then find the optimal threshold for each ring and each ring combination.  The combos I have in mind are pure differences between rows and averaged differences between rows

    #The format for the ring avg column headers are 'Rx' where x is the ring index.  The format for ring count column headers are 'RCx' where x is the ring index
    #for each test, the result will be saved in a dataframe

    mark_list=list(dataframe_input["mark"])
    count_print=np.sum(mark_list)
    count_blank=len(mark_list)-count_print

    thresh_iterators=100

    data=pd.DataFrame(columns=["Test_Name","Thresh","J_Abs","J","Sen","Spec","n_P","n_B"])

    #print "Analysis Started"

    #part 1 is simply running through the rings
    if 1 in test_type:
        for row_index in range(ring_count):
            data_index="R"+str(row_index)
            data_input=list(dataframe_input[data_index])
            thresh,J_abs,J,sen,spec=threshold_finder(data_input,mark_list,thresh_iterators)
            test_name=data_index
            adder_df=pd.DataFrame([[test_name,thresh,J_abs,J,sen,spec,count_print,count_blank]],columns=["Test_Name","Thresh","J_Abs","J","Sen","Spec","n_P","n_B"])
            data=data.append(adder_df)

    #print "Part 1 Done"

    ##part 2 is finding the difference between all the columns
    if 2 in test_type:
        for start_index in range(ring_count):
            for end_index in range(ring_count):
                if end_index>start_index:
                    pos_data_index="R"+str(end_index)
                    pos_data=list(dataframe_input[pos_data_index])
                    neg_data_index="R"+str(start_index)
                    neg_data=list(dataframe_input[neg_data_index])
                    diff_data=list(np.array(pos_data)-np.array(neg_data))
                    thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                    test_name=pos_data_index+"_minus_"+neg_data_index
                    adder_df=pd.DataFrame([[test_name,thresh,J_abs,J,sen,spec,count_print,count_blank]],columns=["Test_Name","Thresh","J","J_Abs","Sen","Spec","n_P","n_B"])
                    data=data.append(adder_df)

    #print "Part 2 Done"

    ###part 3 is doing a weighted average between all combinations of consecutive columns
    if 3 in test_type:    
        neg_index=0
        for start_range_index in range(1,ring_count-1):
            for number_of_columns in range(2,ring_count-start_range_index+2):
                for start_range_fin in range(start_range_index,ring_count-number_of_columns+1):
                    data_list=[]
                    weight_list=[]
                    for col in range(number_of_columns):
                        data_name="R"+str(col+start_range_fin)
                        data_list.append(list(dataframe_input[data_name]))
                        weight_name="RC"+str(col+start_range_fin)
                        weight_list.append(list(dataframe_input[weight_name]))
                    avg_col=weighted_average(data_list,weight_list)

                    neg_data_index="R"+str(neg_index)
                    neg_data=list(dataframe_input[neg_data_index])

                    diff_data=list(np.array(avg_col)-np.array(neg_data))
                    thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                    test_name="R"+str(start_range_fin)+"_to_R"+str(start_range_fin+number_of_columns-1)+"__minus__"+"R"+str(neg_index)
                    adder_df=pd.DataFrame([[test_name,thresh,J_abs,J,sen,spec,count_print,count_blank]],columns=["Test_Name","Thresh","J","J_Abs","Sen","Spec","n_P","n_B"])
                    data=data.append(adder_df)

    #part 4 is finding the difference between all the columns.  By default compares R and RC columns
    if 4 in test_type:    
        for start_index in range(ring_count):
            for end_index in range(ring_count):
                pos_data_index="R"+str(end_index)
                pos_data=list(dataframe_input[pos_data_index])
                neg_data_index="RC"+str(start_index)
                neg_data=list(dataframe_input[neg_data_index])
                diff_data=list(np.array(pos_data)-np.array(neg_data))
                thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                test_name=pos_data_index+"_minus_"+neg_data_index
                adder_df=pd.DataFrame([[test_name,thresh,J_abs,J,sen,spec,count_print,count_blank]],columns=["Test_Name","Thresh","J","J_Abs","Sen","Spec","n_P","n_B"])
                data=data.append(adder_df)


    if 5 in test_type:
        for start_index in range(ring_count):
            for end_index in range(ring_count):
                if end_index>start_index:
                    bin_count=55
                    for bin_count_start in range(bin_count):
                        for bin_count_end in range(bin_count):       
                            pos_data_index="R"+str(end_index)+"b"+str(bin_count_end)
                            pos_data=list(dataframe_input[pos_data_index])
                            neg_data_index="R"+str(start_index)+"b"+str(bin_count_start)
                            neg_data=list(dataframe_input[neg_data_index])
                            diff_data=list(np.array(pos_data)-np.array(neg_data))
                            thresh,J_abs,J,sen,spec=threshold_finder(diff_data,mark_list,thresh_iterators)
                            test_name=pos_data_index+"_minus_"+neg_data_index
                            adder_df=pd.DataFrame([[test_name,thresh,J_abs,J,sen,spec,count_print,count_blank]],columns=["Test_Name","Thresh","J","J_Abs","Sen","Spec","n_P","n_B"])
                            data=data.append(adder_df)
    if 6 in test_type:
        for row_index in range(ring_count):
            bin_count=55
            for bin in range(bin_count):
                data_index="R"+str(row_index)+"b"+str(bin)
                #print data_index
                data_input=list(dataframe_input[data_index])
                thresh,J_abs,J,sen,spec=threshold_finder(data_input,mark_list,thresh_iterators)
                test_name=data_index
                adder_df=pd.DataFrame([[test_name,thresh,J_abs,J,sen,spec,count_print,count_blank]],columns=["Test_Name","Thresh","J_Abs","J","Sen","Spec","n_P","n_B"])
                data=data.append(adder_df)
    #data.to_csv("dataframe_export2.csv")

    #print data
    return data


def pk_dataframe_filter(dataframe_input,formulation,recipe,modification):
    df=copy.copy(dataframe_input)

    #KEEP ROWS THAT MATCH THE FORMULATION OR BLANK
    if not formulation=='skip':
        df=df.loc[df['Formulation'].str.contains(formulation)]
    #print "prog1"
    #KEEP ROWS THAT MATCH THE RECIPE
    if not recipe[0]=='skip':
        df=df.loc[df['Size'].isin([recipe[0]])]
    if not recipe[1]=='skip':
        df=df.loc[df['Brand'].isin([recipe[1]])]
    if not recipe[2]=='skip':
        df=df.loc[df['Location'].isin([recipe[2]])]
    #print "prog2"
    #KEEP ROWS THAT MATCH THE MODIFICATION
    if not modification=='skip':
        #make lowercase and uppercase options
        df=df.loc[df['AP'].isin([modification])]

    return df

def pk_redundancy_checker(dataframe_input,threshold_dictionary,red_value,test_type=[1]):
    #The first step is to convert the threshold_dictionary to a vector of vectors, each of which are 1 or 0, based on comparing the guess to the mark
    mark_list=list(dataframe_input["mark"])
    path_list=list(dataframe_input["Path"])
    count_print=np.sum(mark_list)
    count_blank=len(mark_list)-count_print
    guess_matrix=[]
    diff_matrix=[]
    key_list=[]
    #print threshold_dictionary

    for key in threshold_dictionary:
        key_list.append(key)
        #print key
        if 'minus' in key:
            col_list=key.split("_")
            col_positive=col_list[0]
            col_negative=col_list[2]
            positive_data=list(dataframe_input[col_positive])
            negative_data=list(dataframe_input[col_negative])
            diff_data=list(np.array(positive_data)-np.array(negative_data))
            thresh=threshold_dictionary[key]
            J,guess_vector,accuracy_vector=threshold_tester(diff_data,mark_list,thresh)
            diff_vector=list(np.array(diff_data)-thresh)
            guess_matrix.append(guess_vector)
            diff_matrix.append(diff_vector)
        else:
            data=list(dataframe_input[key])
            thresh=threshold_dictionary[key]
            J,guess_vector,accuracy_vector=threshold_tester(data,mark_list,thresh)
            diff_vector=list(np.array(data)-thresh)
            guess_matrix.append(guess_vector)
            diff_matrix.append(diff_vector)

    #iterate through accuracy_matrix with various levels of redundancy

    # Redundancy method
    guess_matrix_cols=range(len(key_list))
    redundancy_list=np.linspace(0.5,len(guess_matrix_cols)-0.5,len(guess_matrix_cols))
            
    redundancy=red_value
    redundancy_guess_list=[]
    for row in range(len(guess_matrix[0])):
        true_guess_sum=0
        for col in guess_matrix_cols:
            true_guess_sum+=guess_matrix[col][row]
        if true_guess_sum>redundancy:
            redundancy_guess_list.append(1)
        else:
            redundancy_guess_list.append(0)
    #Calculate sen, spec, and J
    J,sen,spec=J_from_vectors(redundancy_guess_list,mark_list)
    #print J,
    #print ",",
    #print sen,
    #print ",",
    #print spec,
    #print ",",
    #print guess_matrix_cols
    max_J=J
    best_sen=sen
    best_spec=spec
    best_red=redundancy
    best_guess_mat=guess_matrix_cols
    best_guess_list=redundancy_guess_list


    #if 2 in test_type:
    #    for iter in range(1,len(key_list)+1):
    #        #print iter
    #        for val in it.combinations(range(0,len(key_list)),iter):
    #            guess_matrix_cols=list(val)
    #            sum_vector=[]
    #            for row in range(len(guess_matrix[0])):
    #                col_sum=0
    #                for col in guess_matrix_cols:
    #                    col_sum+=diff_matrix[col][row]
    #                sum_vector.append(col_sum)
    #            #Calculate sen, spec, and J
    #            thresh,J_abs,J,sen,spec=threshold_finder(sum_vector,mark_list,20)
    #            print J,
    #            print ",",
    #            print sen,
    #            print ",",
    #            print spec,
    #            print ",",
    #            print thresh,
    #            print ",",
    #            print guess_matrix_cols
    #            if J>max_J:
    #                max_J=J
    #                best_sen=sen
    #                best_spec=spec
    #                best_red=thresh
    #                best_guess_mat=guess_matrix_cols
    #                J,guess_vector,accuracy_vector=threshold_tester(sum_vector,mark_list,thresh)
    #                best_guess_list=guess_vector
                        
  #  print best_guess_mat

    #save the winning columns and thresholds
    best_guess_mat2=[]
    for iter in best_guess_mat:
        key=key_list[iter]
        value=threshold_dictionary[key]
        best_guess_mat2.append([key,value])

    fail_pic_list=[]
    for iter in range(len(best_guess_list)):
        if not best_guess_list[iter]==mark_list[iter]:
            fail_pic_list.append(path_list[iter])
            #print path_list[iter]

    #for fail_pic_dir in fail_pic_list:
    #    fix_path=fail_pic_dir.replace('\\',"//")
    #    shutil.copy2(fail_pic_dir,'bad_pics')

    return max_J,best_sen,best_spec,count_blank,count_print,best_red,best_guess_mat2,fail_pic_list

def pk_redundancy_tester(dataframe_input,threshold_dictionary,test_type=[1]):
    #The first step is to convert the threshold_dictionary to a vector of vectors, each of which are 1 or 0, based on comparing the guess to the mark
    mark_list=list(dataframe_input["mark"])
    path_list=list(dataframe_input["Path"])
    count_print=np.sum(mark_list)
    count_blank=len(mark_list)-count_print
    guess_matrix=[]
    diff_matrix=[]
    key_list=[]
    #print threshold_dictionary

    for key in threshold_dictionary:
        key_list.append(key)
        #print key
        if 'minus' in key:
            col_list=key.split("_")
            col_positive=col_list[0]
            col_negative=col_list[2]
            positive_data=list(dataframe_input[col_positive])
            negative_data=list(dataframe_input[col_negative])
            diff_data=list(np.array(positive_data)-np.array(negative_data))
            thresh=threshold_dictionary[key]
            J,guess_vector,accuracy_vector=threshold_tester(diff_data,mark_list,thresh)
            diff_vector=list(np.array(diff_data)-thresh)
            guess_matrix.append(guess_vector)
            diff_matrix.append(diff_vector)
        else:
            data=list(dataframe_input[key])
            thresh=threshold_dictionary[key]
            J,guess_vector,accuracy_vector=threshold_tester(data,mark_list,thresh)
            diff_vector=list(np.array(data)-thresh)
            guess_matrix.append(guess_vector)
            diff_matrix.append(diff_vector)

    #iterate through accuracy_matrix with various levels of redundancy

    # Redundancy method
    max_J=0
    best_sen=0
    best_spec=0
    best_red=0
    best_guess_mat=[]
    best_guess_list=[]
    #print key_list
    if 1 in test_type:
      #  for iter in range(1,len(key_list)+1):
        for iter in range(1,4):
            #print iter
            for val in it.combinations(range(0,len(key_list)),iter):
                guess_matrix_cols=list(val)
                redundancy_list=np.linspace(0.5,len(guess_matrix_cols)-0.5,len(guess_matrix_cols))
            
                for redundancy in redundancy_list:
                    redundancy_guess_list=[]
                    for row in range(len(guess_matrix[0])):
                        true_guess_sum=0
                        for col in guess_matrix_cols:
                            true_guess_sum+=guess_matrix[col][row]
                        if true_guess_sum>redundancy:
                            redundancy_guess_list.append(1)
                        else:
                            redundancy_guess_list.append(0)
                    #Calculate sen, spec, and J
                    J,sen,spec=J_from_vectors(redundancy_guess_list,mark_list)
                    #print J,
                    #print ",",
                    #print sen,
                    #print ",",
                    #print spec,
                    #print ",",
                    #print guess_matrix_cols
                    if J>max_J:
                        max_J=J
                        best_sen=sen
                        best_spec=spec
                        best_red=redundancy
                        best_guess_mat=guess_matrix_cols
                        best_guess_list=redundancy_guess_list
    if 2 in test_type:
        for iter in range(1,len(key_list)+1):
            #print iter
            for val in it.combinations(range(0,len(key_list)),iter):
                guess_matrix_cols=list(val)
                sum_vector=[]
                for row in range(len(guess_matrix[0])):
                    col_sum=0
                    for col in guess_matrix_cols:
                        col_sum+=diff_matrix[col][row]
                    sum_vector.append(col_sum)
                #Calculate sen, spec, and J
                thresh,J_abs,J,sen,spec=threshold_finder(sum_vector,mark_list,20)
                print J,
                print ",",
                print sen,
                print ",",
                print spec,
                print ",",
                print thresh,
                print ",",
                print guess_matrix_cols
                if J>max_J:
                    max_J=J
                    best_sen=sen
                    best_spec=spec
                    best_red=thresh
                    best_guess_mat=guess_matrix_cols
                    J,guess_vector,accuracy_vector=threshold_tester(sum_vector,mark_list,thresh)
                    best_guess_list=guess_vector
                        
    #print best_guess_mat

    #save the winning columns and thresholds
    best_guess_mat2={}
    for iter in best_guess_mat:
        
        key=key_list[iter]
        value=threshold_dictionary[key]
        best_guess_mat2[key]=value
        #best_guess_mat2.append([key,value])

    fail_pic_list=[]
    for iter in range(len(best_guess_list)):
        if not best_guess_list[iter]==mark_list[iter]:
            fail_pic_list.append(path_list[iter])
            #print path_list[iter]

    #for fail_pic_dir in fail_pic_list:
    #    fix_path=fail_pic_dir.replace('\\',"//")
    #    shutil.copy2(fail_pic_dir,'bad_pics')

    return max_J,best_sen,best_spec,count_blank,count_print,best_red,best_guess_mat2,fail_pic_list


def print_exif_UC(image_path):
    img=PIL.Image.open(image_path)
    exif={
         PIL.ExifTags.TAGS[k]:v
         for k, v in img._getexif().items()
         if k in PIL.ExifTags.TAGS
         }
    print exif
    return exif['UserComment']

def print_exif_DT(image_path):
    img=PIL.Image.open(image_path)
    exif={
         PIL.ExifTags.TAGS[k]:v
         for k, v in img._getexif().items()
         if k in PIL.ExifTags.TAGS
         }

    return exif['DateTimeOriginal']
   
def logistic_percent_calculator(coeffs,intercept,parameters):
    if not len(coeffs)==len(parameters):
        return -1
    else:
        t_sum=0
        for iter in range(0,len(coeffs)):
            t_sum+=coeffs[iter]*parameters[iter]
        t_sum+=intercept
        log_regression=float(1)/float(1+math.exp(t_sum*-1))
        return log_regression

def import_and_sort_csv(csv_file,return_number,sort_index):
    super_list=[]
    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            try:
                float_row=[float(i) for i in row]

                min_val=np.min([float_row[0],float_row[1]])
                max_stats=np.max([float_row[0:5]])
            
                float_row.insert(0,min_val)

                if min_val>0 and min_val<=1 and max_stats<=1:
                    super_list.append(float_row)
            except:
                null=1
    
    super_list=sorted(super_list,key=itemgetter(sort_index),reverse=True)
    print type(super_list[0][0])
    for i in range(0,return_number):
        print super_list[i]

    return 0

def save_failed_images(dataframe_input,mark_list,sum_list,best_redundancy):
    #save failed images
    falsenegative_fail_image_paths=[]
    falsepositive_failed_image_paths=[]
    for save_fail_index in range(len(mark_list)):
        if mark_list[save_fail_index]==1 and sum_list[save_fail_index]<best_redundancy:
            falsenegative_fail_image_paths.append(dataframe_input.at[save_fail_index,'path'])
        if mark_list[save_fail_index]==0 and sum_list[save_fail_index]>best_redundancy:
            falsepositive_failed_image_paths.append(dataframe_input.at[save_fail_index,'path'])

    #copy failed images to their destinations
    for fn_path in falsenegative_fail_image_paths:
        path_split=fn_path.split("\\")
        file_name=path_split[-1]
        file_move=os.path.join("modeler_badPics//falsenegative",file_name)
        shutil.copyfile(fn_path,file_move)
    for fp_path in falsepositive_failed_image_paths:
        path_split=fp_path.split("\\")
        file_name=path_split[-1]
        file_move=os.path.join("modeler_badPics//falsepositive",file_name)
        shutil.copyfile(fp_path,file_move)
    return

def weighted_average(input_data, input_weights):
    #data should be a list of lists.
    if not len(input_data)==len(input_weights):
        raise ValueError('Input mark and data are not of equal size')
    #print input_data
    #print input_weights
    export_data=[]
    for row in range(len(input_data[0])):
        data_list=[]
        weight_list=[]
        for col in range(len(input_data)):
            data_list.append(input_data[col][row])
            weight_list.append(input_weights[col][row])
        #print data_list
        #print weight_list
        #print len(data_list)
        #print len(weight_list)
        if np.sum(weight_list)==0:
            export_avg=np.average(data_list)
        else:
            export_avg=np.average(data_list,weights=weight_list)
        export_data.append(export_avg)
    return export_avg

def set_column_sequence(dataframe, seq, front=True): #https://stackoverflow.com/questions/12329853/how-to-rearrange-pandas-column-sequence/23741704
    '''Takes a dataframe and a subsequence of its columns,
       returns dataframe with seq as first columns if "front" is True,
       and seq as last columns if "front" is False.
    '''
    cols = seq[:] # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            if front: #we want "seq" to be in the front
                #so append current column to the end of the list
                cols.append(x)
            else:
                #we want "seq" to be last, so insert this
                #column in the front of the new column list
                #"cols" we are building:
                cols.insert(0, x)
    return dataframe[cols]