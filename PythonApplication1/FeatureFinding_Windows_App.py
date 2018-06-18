import glob
import os
import json
from glob import glob
import pandas as pd
import copy
import math
import numpy as np
import matplotlib.pyplot as plt


def FF_icon():

    start_dir = raw_input('Enter starting directory for the csv files of interest:')
    start_json_dir = raw_input('Enter starting directory for the json files of interest:')
    outputfilename = (raw_input('Enter the name of the xlsx file that will be outputted by this program:'))
    outputfile_loc = raw_input('Enter the directory where {}{} will be saved:'.format(outputfilename, '.xlsx'))
    outputfile_dir = '{}{}{}{}'.format(outputfile_loc,'/',outputfilename,'.xlsx')
    point_count = input('Enter the number of regions of interest that this particular test analyzes:')
    test_index = []

    #start_dir = "F:\\\Developer\\csvExports\\180608 JL Exports\\180606"
    #start_json_dir = "F:\\\Developer\jsonExports\Juby Lee"
    #outputfilename = "TestOutput"
    #outputfile_loc = "C:\\\Users\\User\\Desktop\\temp"
    #FileExtension = ".xlsx"
    #filename = "F:\\\Developer\csvExports\ 180608 JL Exports\180606\JL FF0 superanalysis_605'705'810'-10'22.csv"
    #outputfile_dir = "{}{}{}{}".format(outputfile_loc, '\\' ,outputfilename , FileExtension)
    #SaveLocation = (outputfile_loc + os.path.splitext(filename[len(start_dir):])[0] + '.png')
    #print SaveLocation
    #point_count = 10

    line_by_line_check = 0
    plot_data_switch = 1
    export_data_switch = 1

    files = []
    json_files = []
    pattern = "*.csv"
    json_pattern = "*.json"
    print 'prog1'

    for dir, _, _ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir, pattern)))
    for dir, _, _ in os.walk(start_json_dir):
        json_files.extend(glob(os.path.join(dir, json_pattern)))

    std_dict = {}
    for json_file in json_files:
        with open(json_file) as json_data:
            adder_dict = json.load(json_data)
        std_dict.update(adder_dict)

    # Delete keys that don't the the correct number of points
    for key in std_dict.keys():
        if not len(std_dict[key]) == point_count:
            del std_dict[key]

    # Prepare export dictionary
    export_list = []

    for file in files:

        filename = file.rpartition("\\")[2]
        # Import csv to dataframe only once

        if (line_by_line_check):
            print filename

        df = pd.read_csv(file, header=0, error_bad_lines=False, warn_bad_lines=False)

        if line_by_line_check:
            print "ORIGINAL DATAFRAME: "
            print df

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if line_by_line_check:
            print "DROP UNNAMED: "
            print df

        # df=fun.arbitrary_include(df,'path','iP8plus')

        distance_superdict = {}
        distance_filter_superlist = {}
        # print filename
        # print '{} {} {} {}'.format('point','avg','stdev','count')
        export_data = ["mean", "std", "median", "count", "filtered_mean", "filtered_std", "filtered_median",
                       "filtered_count", "outlier_count"]
        summary_dict = {}
        for ff_index in range(point_count):
            output_dict = ff_accuracy(df, std_dict, ff_index)
            mean_dist = output_dict["mean"]
            stdev_dist = output_dict["std"]
            list_dist = output_dict["data"]
            dict_dist = output_dict["data_dict"]
            median_dist = output_dict["median"]
            filtered_data = output_dict["filtered_data"]
            print '{0:2d} {1:3f} {2:4f} {3:5d}'.format(ff_index,mean_dist,stdev_dist,len(list_dist))

            # Save data to export list
            temp_dict = {}
            for key in export_data:
                temp_dict[key] = output_dict[key]
            temp_dict["path"] = file
            temp_dict["filename"] = filename
            temp_dict["test"] = "roi" + str(int(ff_index))
            export_list.append(temp_dict)
            distance_superdict[str(ff_index)] = list_dist
            distance_filter_superlist[str(ff_index)] = filtered_data

            # Save data to summary dict
            if ff_index == 0:
                summary_dict = temp_dict
            else:
                for key in export_data:
                    summary_dict[key] += temp_dict[key]
        if 't_total' in df:
            mean_t = df["t_total"].mean()
            stdev_t = df["t_total"].std()
            count_t = df["t_total"].count()
            time_series = df["t_total"]
            time_series = time_series.dropna()
            # print 'time {0:3f} {1:4f} {2:5d}'.format(mean_t,stdev_t,count_t)

            # save data to export list
            output_dict = summary_statistics(list(time_series))
            temp_dict = {}
            for key in export_data:
                temp_dict[key] = output_dict[key]
            temp_dict["path"] = file
            temp_dict["filename"] = filename
            temp_dict["test"] = "time"
            export_list.append(temp_dict)

        # do summary stuff for the rois
        temp_dict = {}
        for key in export_data:
            summary_dict[key] = summary_dict[key] / float(point_count)
        summary_dict["path"] = file
        summary_dict["filename"] = filename
        summary_dict["test"] = "roisummary"
        # export_list.append(summary_dict)
        summary_dict = {}
        #    for key in dict_dist.keys():
        #        print key

        # print distance_superdict
        # for key in distance_superdict[0].keys():
        #    print key

        if (plot_data_switch):
            for key in distance_superdict:
                data = distance_superdict[key]
                if len(data) > 0:
                    plt.subplot(131)
                    plt.hist(distance_superdict[key], alpha=0.3, label=key, normed=1, histtype='stepfilled')
                    plt.xlabel('raw data')
                    plt.subplot(132)
                    plt.hist(distance_filter_superlist[key], alpha=0.3, label=key, normed=1, histtype='stepfilled')
                    plt.xlabel('outlier filtered data')
                    plt.subplot(133)
                    plt.hist(distance_filter_superlist[key], alpha=0.3, label=key, normed=1, histtype='stepfilled')
                    plt.xlabel('ot_filt data locked rng')
                    plt.xlim(0, 75)
                    plt.ylim(0, 0.25)
                    # label_list.append(key)
            plt.legend()
            plt.subplot(132)
            plt.title(filename)
            # print filename
            # print outputfile_loc
            # print os.path.splitext(filename[len(start_dir):])[0] + '.png'
            # print outputfile_loc + os.path.splitext(filename[len(start_dir):])[0] + '.png'
            #SaveLocationPath = (outputfile_loc + os.path.splitext(file[len(start_dir):])[0] + '.png')
            #ThisFileBeg = outputfile_loc
            #ThisFileMid = os.path.splitext(file[len(start_dir):])[0]
            #ThisFileEnd = '.png'
            #myfile1 = open(outputfile_loc + os.path.splitext(file[len(start_dir):])[0] + '.png', 'wb')
            plt.savefig(outputfile_loc + os.path.splitext(file[len(start_dir):])[0] + '.png')
            plt.clf()
            plt.cla()
            plt.close()
    if (export_data_switch):
        myfile = open(outputfile_dir, 'wb')
        final_df = pd.DataFrame(export_list)
        writer = pd.ExcelWriter(outputfile_dir )
        final_df.to_excel(writer, 'data')
        writer.save()

    return



def path_cleanup(input_path):
    path_edit=copy.copy(input_path)
    path_edit=path_edit.replace("\\","")
    path_edit=path_edit.replace("//","")
    path_edit=path_edit.replace(" ","")
    path_edit=path_edit.lower()
    path_edit=path_edit.replace("f:","")
    path_edit=path_edit.replace("c:usersgtthoresiliosync","")
    path_edit=path_edit.replace(":","")
    return path_edit




def euclidian_distance(reference_point,test_point):
    if not len(reference_point)==len(test_point):
        raise ValueError('Reference and Tests Points not the same length in euclidian distance calculation')

    sum_to_square_root=0
    for point_index in range(len(reference_point)):
        sum_to_square_root+=math.pow(reference_point[point_index]-test_point[point_index],2)
    distance=math.sqrt(sum_to_square_root)
    return distance



def ff_accuracy(dataframe_input,standard_map,ff_index):
    #clean up 'path' column of dataframe_input and keys in standard_map to eliminate '\','/' and make everything lowercase
    data=copy.copy(dataframe_input)
    data=data.reset_index(drop=True)
    data['path_clean'] = data.apply(lambda row: path_cleanup(row['path']), axis=1)
    data['path_clean']=data['path_clean'].astype(str)
    data=data.set_index('path_clean')

    map_clean={}
    for key in standard_map:
        key_clean=path_cleanup(key)
        key_clean=str(key_clean)
        map_clean[key_clean]=standard_map[key]

    #print map_clean
    #print data.index
    #now, go through all standard keys and calculate euclidian distances.  Save it as a vector of data, a dictionary, and summary stats
    distance_dict={}
    distance_list=[]
    for key in map_clean:
        #print key
        if key in data.index:
            #print "###################KEY FOUND####################"
            std_x=map_clean[key][ff_index][0]
            std_y=map_clean[key][ff_index][1]
            ff_column='ff_point'+str(int(ff_index))
            test_x=data.at[key,ff_column+'x']
            test_y=data.at[key,ff_column+'y']
            if test_x==-1 or test_y==-1: continue
            distance=euclidian_distance([std_x,std_y],[test_x,test_y])
            #print data.at[key,'path']
            if not math.isnan(distance):
                distance_dict[data.at[key,'path']]=distance
                distance_list.append(distance)

    return_dict=summary_statistics(distance_list)
    return_dict["data_dict"]=distance_dict

    return return_dict


def summary_statistics(data):
    mean_data=np.average(data)
    stdev_data=np.std(data)
    median_data=np.median(data)
    filtered_list=reject_outliers(data,3)
    filtered_mean=np.average(filtered_list)
    filtered_stdev=np.std(filtered_list)
    filtered_median=np.median(filtered_list)
    outlier_count=len(data)-len(filtered_list)

    #return data as a dict
    return_dict={}
    return_dict["mean"]=mean_data
    return_dict["std"]=stdev_data
    return_dict["median"]=median_data
    return_dict["data"]=data
    return_dict["count"]=len(data)
    return_dict["filtered_data"]=filtered_list
    return_dict["filtered_mean"]=filtered_mean
    return_dict["filtered_std"]=filtered_stdev
    return_dict["filtered_median"]=filtered_median
    return_dict["filtered_count"]=len(filtered_list)
    return_dict["outlier_count"]=outlier_count

    return return_dict


def reject_outliers(data, m = 2.): #pulled from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    data_new=[]
    for count,val in enumerate(s):
        if val<m: data_new.append(data[count])
    return data_new


N = FF_icon()
