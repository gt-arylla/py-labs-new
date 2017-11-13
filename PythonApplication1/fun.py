#functions
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import copy
import itertools as it
import csv

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
                if test_type==0:#one photo method
                    if sum(mark_pred_mini_list)>=0.5:
                        print_pred_counter+=1
                if test_type==20:
                    if sum(mark_pred_mini_list)==2:
                        print_pred_counter+=1
                if test_type==21:
                    max_confidence=max(confidence_mini_list)
                    for k in range(0,len(confidence_mini_list)):
                        if confidence_mini_list[k]==max_confidence:
                            pred_val_final=mark_pred_mini_list[k]
                    if pred_val_final==1:
                        print_pred_counter+=1
                if test_type==30:#three pics, best two out of three of the mark_pred wins
                    if sum(mark_pred_mini_list)>1.5:
                        print_pred_counter+=1
                if test_type==31:#three pics, best two out of three of the mark_pred wins
                    if sum(mark_pred_mini_list)>2.5:
                        print_pred_counter+=1
                if test_type==32:
                    max_confidence=max(confidence_mini_list)
                    for k in range(0,len(confidence_mini_list)):
                        if confidence_mini_list[k]==max_confidence:
                            pred_val_final=mark_pred_mini_list[k]
                    if pred_val_final==1:
                        print_pred_counter+=1
                if test_type==33:
                    avg_pred_val=sum(mark_pred_mini_list)/float(len(mark_pred_mini_list))
                    if avg_pred_val==0:
                        null=1;
                    elif avg_pred_val==1:
                        print_pred_counter+=1
                    else: #split into blank and print 
                        print_vec=[]
                        blank_vec=[]
                        for k in range(0,len(mark_pred_mini_list)):
                            if mark_pred_mini_list[k]==0:
                                blank_vec.append(confidence_mini_list[k])
                            elif mark_pred_mini_list[k]==1:
                                print_vec.append(confidence_mini_list[k])
                        blank_conf_avg=sum(blank_vec)/float(len(blank_vec))
                        print_conf_avg=sum(print_vec)/float(len(print_vec))
                        if print_conf_avg>blank_conf_avg:
                            print_pred_counter+=1

                if test_type==4:#three pics, best two out of three of the mark_pred wins
                    if sum(mark_pred_mini_list)>1.5:
                        print_pred_counter+=1
                if test_type==40:#three pics, best two out of three of the mark_pred wins
                    if sum(mark_pred_mini_list)>2.5:
                        print_pred_counter+=1
                if test_type==41:#three pics, best two out of three of the mark_pred wins
                    if sum(mark_pred_mini_list)>3.5:
                        print_pred_counter+=1
                if test_type==42:#three pics, best two out of three of the mark_pred wins
                    max_confidence=max(confidence_mini_list)
                    for k in range(0,len(confidence_mini_list)):
                        if confidence_mini_list[k]==max_confidence:
                            pred_val_final=mark_pred_mini_list[k]
                    if pred_val_final==1:
                        print_pred_counter+=1
                if test_type==43:
                    avg_pred_val=sum(mark_pred_mini_list)/float(len(mark_pred_mini_list))
                    if avg_pred_val==0:
                        null=1;
                    elif avg_pred_val==1:
                        print_pred_counter+=1
                    else: #split into blank and print 
                        print_vec=[]
                        blank_vec=[]
                        for k in range(0,len(mark_pred_mini_list)):
                            if mark_pred_mini_list[k]==0:
                                blank_vec.append(confidence_mini_list[k])
                            elif mark_pred_mini_list[k]==1:
                                print_vec.append(confidence_mini_list[k])
                        blank_conf_avg=sum(blank_vec)/float(len(blank_vec))
                        print_conf_avg=sum(print_vec)/float(len(print_vec))
                        if print_conf_avg>blank_conf_avg:
                            print_pred_counter+=1
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

#def logistic_regression(csv_file,x_cols,row_keep, printer)

