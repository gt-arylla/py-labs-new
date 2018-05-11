#
#import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import itertools as it
import scipy


#Plot data from input CSV, organized according to the FAF C++ code
def FAF_plotter(csv_file,x_col_in,y_col_in,s_col,yaxis=[],xaxis=[]):
    #Define Variables
    #x_col=[5]
    #y_col=[7]
    ##z_col=[0]
    #s_col=[0,1,2,3,4] #first value is the z value
    
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
    print 'Import Data...'
    if len(s_col)==1:
        dummy_add=True
        s_col.append(col_count)
    else:
        dummy_add=False
    if dummy_add:
        col_count+=1
    all_data = np.zeros(shape=(row_count - 1,col_count))
    row_count = 0
    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        counter = 0
        for row in readCSV:
            if counter > 0:
                try:
                    if dummy_add:
                        row.append(0)
                    all_data[row_count,:] = row
                    row_count+=1
                except:
                    print "Row Import Failed"
            else:
                headers = row
            counter+=1
    print all_data

    '''
    #Pull out unique values for s_col
    print 'Pull out unique values for s_col...'
    unique_vals=[];
    for row in np.arange(row_count):
        for i in np.arange(len(s_col)):
            if row==0:
                unique_vals.append([all_data[row,i]])
            else:
                printer=False
                for j in np.arange(len(unique_vals[i])):
                    if unique_vals[i][j]!=all_data[row,i]:
                        printer=True
                if printer:
                    unique_vals[i].append(all_data[row,i]);
    print unique_vals
    '''
    unique_vals = []
    for i in s_col:
        unique_list = np.unique(all_data[:,i])
        unique_vals.append(unique_list)

    print unique_vals


    #Construct nested lists to hold unique values and z values
    #nest=[]
    #for i in np.arange(len(unique_vals)):
    #    for j in np.arange(unique_vals[i]):

    #Construct vector of unique values, saved as strings
    print 'Construct vector of unique values, saved as strings...'
    unique_vals_string = []
    for i in np.arange(len(unique_vals)):
        unique_vals_string.append([])
        for j in np.arange(len(unique_vals[i])):
            unique_vals_string[i].append(str(unique_vals[i][j]))

    #construct super vector with all possible combinations of unique vals
    print 'construct super vector with all possible combinations of unique vals...'
    print unique_vals_string
    super_list = [" ".join(i) for i in it.product(*unique_vals_string)]

    #for items in it.product(*unique_vals_string):
    #    super_list.append(items)
    
    print super_list

    if len(x_col_in) == 1:
        x_vec = [x_col_in[0],x_col_in[0] + 1]
    else:
        x_vec = [x_col_in[0],x_col_in[1] + 1]
    if len(y_col_in) == 1:
        y_vec = [y_col_in[0],y_col_in[0] + 1]
    else:
        y_vec = [y_col_in[0],y_col_in[1] + 1]
        
    for x_col in np.arange(x_vec[0],x_vec[1]):
        for y_col in np.arange(y_vec[0],y_vec[1]):

            #construct holding dictionary
            print 'construct holding dictionary...'
            values_dictionary = {}
            for i in np.arange(len(super_list)):
                values_dictionary[super_list[i]] = [[],[]]
            print values_dictionary.keys()
            #add data to holding dictionary
            print 'add data to holding dictionary...'
            for row in np.arange(row_count):
                key = ''
                for s_index in s_col:
                    key+=str(all_data[row,s_index]) + ' '
            
                key = key[:-1]
                #print key
                values_dictionary[key][0].append(all_data[row,x_col])
                values_dictionary[key][1].append(all_data[row,y_col])

            #plot data
            print 'plot data...'
            '''
            unique_vals_string_no_Z=[]
            for i in np.arange(1,len(unique_vals)):
                unique_vals_string_no_Z.append([]);
                for j in np.arange(len(unique_vals[i])):
                    unique_vals_string_no_Z[i].append(str(unique_vals[i][j]))
        '''

            #Construct vector of unique values, saved as strings
            print 'Construct vector of unique values, saved as strings...'
            #unique_vals_string_no_Z=[]
            #for i in np.arange(1,len(unique_vals)):
            #    unique_vals_string_no_Z.append([]);
            #    for j in np.arange(len(unique_vals[i])):
            #        print str(i)+','+str(j)
            #        unique_vals_string_no_Z.append(str(unique_vals[i][j]))
            unique_vals_string_no_Z = []
            for i in np.arange(1,len(unique_vals_string)):
                unique_vals_string_no_Z.append(unique_vals_string[i])
            print unique_vals_string_no_Z
            super_list_no_Z = [" ".join(i) for i in it.product(*unique_vals_string_no_Z)]
            #for items in it.product(*unique_vals_string_no_Z):
            #    super_list_no_Z.append(items)
            print super_list_no_Z
            for i in super_list_no_Z:
                plt.figure(1)
                for z_val in unique_vals_string[0]:
                    #z_val_str=str(z_val)
                    key = z_val + ' ' + i
                    print key
                    plt.plot(values_dictionary[z_val + ' ' + i][0],values_dictionary[z_val + ' ' + i][1],label=z_val,alpha=0.5,marker='.',linestyle = 'None',ms=4)
                plt.title(i)
                plt.legend(loc='best')
                axes = plt.gca()

                if len(yaxis) == 2:
                    ymin = yaxis[0]
                    ymax = yaxis[1]
                    axes.set_ylim([ymin,ymax])

                if len(xaxis) == 2:
                    xmin = xaxis[0]
                    xmax = xaxis[1]
                    axes.set_xlim([xmin,xmax])
                plt.xlabel(headers[x_col])
                plt.ylabel(headers[y_col])
        
                #plt.show()
                print "Saving..."
                plt.savefig(headers[y_col]+" vs "+headers[x_col]+"--"+i + '.jpg', format='jpg', dpi=200)
                plt.clf()




    #Cause I don't know how to make nested empy list, append everything to one
    #list
    #nest=[]
    #info=[]
    #for i in unique_vals[0]:
    #    for j in unique_vals[1]:
    #        for k in unique_vals[2]:
    #            for l in unique_vals[3]:
    #                for m in unique_vals[4]:
    #                    info.append([i,j,k,l,m])
    #                    nest.append([])

    #Add data to nest
    #for row in np.arange(row_count):
    #    for i in np.arange(len(unique_vals)):
    #        for i_val in np.arange(len(unique_vals[i])):
    #            if all_data[row,s_col[i]]==val

    return

def cotsu_plotter(csv_file):
    df=pd.read_csv("G://Google Drive//Datasets//180122_cOtsu_till_App36.csv",header=0)
    df['PhotoSet']=df['PhotoSet'].astype(str)

    df_train=df.loc[df['Category'].str.contains("Train1")]
    df_test=df.loc[df['Category'].str.contains("Test1|Test3")==True]
    df_train_101=df_train.loc[df['PhotoSet'].str.contains("101")]
    df_train_102=df_train.loc[df['PhotoSet'].str.contains("102")]
    df_test_101=df_test.loc[df['PhotoSet'].str.contains("101")]
    df_test_102=df_test.loc[df['PhotoSet'].str.contains("102")]

    bins=["WH_White","WH_Black","LG_White","LG_Black","MG_White","MG_Black","DG_White","DG_Black","BK_White","BK_Black","R_White","R_Black","O_White","O_Black","Y_White","Y_Black","SG_White","SG_Black","G_White","G_Black","TU_White","TU_Black","CY_White","CY_Black","AZ_White","AZ_Black","B_White","B_Black","PU_White","PU_Black","MA_White","MA_Black","PI_White","PI_Black","BC_White","BC_Black"]

    for bin in bins:
        for PhotoSet in [101,102]:
            if PhotoSet==101:
                data_test=df_test_101[bin]
                data_train=df_train_101[bin]
            if PhotoSet==102:
                data_test=df_test_102[bin]
                data_train=df_train_102[bin]
            #f,p=scipy.stats.f_oneway(data_test,data_train)
            t,p2=scipy.stats.ttest_ind(data_test,data_train)
            print "Bin, PhotoSet, f, p:"+bin +","+str(PhotoSet)+","+str(t)+","+str(p2)+","+str(max(data_test))+','+str(max(data_train))
            #if PhotoSet==101:
            #    plt.subplot(121)
            #if PhotoSet==102:
            #    plt.subplot(122)
            #plt.hist([data_test,data_train],10,alpha=0.3,normed=1,histtype='stepfilled')
            #plt.title(bin+" "+str(PhotoSet))
            #plt.legend(['test','train'])
            #plt.xlabel("p=%s" %p2)
            
            #if PhotoSet==102:
            #    #plt.show()
                
            #    plt.savefig(bin+".jpg")
            #    plt.clf()
def pivot_histogram(csv_file,header,data):
    df=pd.read_csv(csv_file,header=0)

    print df

    del df['DateTime']
    #del df['Is_White']
    #del df['Is_Square']

    print df

    #pivot_table=df.pivot(df,index='Key',columns='Tag',values='Match')

    #dfs2 = [rows for _, rows in df.groupby('Category')]
    dfs2=[df]
    for internal_df in dfs2:
        dfs=[rows for _, rows in internal_df.groupby('Category')]
        data_cols=['Match','Duration']
        count=1
        titl=""
        for data_col in data_cols:

            dfs_data=[]
            legend_list=[]
            for internal_df in dfs:
                #print internal_df[data_col].mean
                dfs_data.append(internal_df[data_col])
                legend_list.append(internal_df.iloc[1,0])
            #print pivot_table
            titl=internal_df.iloc[1,1]+" "+data_col
            plt.subplot(len(data_cols),1,count)
            plt.hist(dfs_data,100,alpha=0.3,normed=1,histtype='stepfilled')
            plt.title(titl)
            #plt.xlim([0,255])
            plt.legend(legend_list)
            count+=1
       
        for internal_df in dfs:
            print internal_df.iloc[1,0]
            print internal_df.iloc[1,1]
            print internal_df.describe()
        plt.show()
        plt.savefig(titl+".jpg")
        plt.clf()
    