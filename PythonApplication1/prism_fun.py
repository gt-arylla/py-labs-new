#Imports
import json,fun, os
import itertools as it
import shutil

def transcript_to_dict(roi_index):
	#finds the transcript file that corresponds to the input roi_index
	#makes a dictionary where the key is the filename, and the value is the score

    roi_string = str(int(roi_index))

    #prep empty coeffs vector
    coeffs = [0.0 for i in range(630)]

    #populate coeffs vector with data.  For now we don't include the intercept,
    #so we skip a=0 and everything else gets put in an index back
    for line in open("coeffs" + "_" + roi_string + ".txt"):
        a,v = line.strip().split()
        if int(a) == 0: continue
        coeffs[int(a) - 1] = float(v)

    thresh = 0

    for line in open("testperf" + "_" + roi_string + ".txt"):
        try:
            J = json.loads(line)
        except:
            continue
    thresh = J["test_ThreshScore"]

    output_dict = {}
    for line in open("transcript" + "_" + roi_string + ".txt"):
        pieces = line.strip().split(";") 
        #class
        cl = int(pieces[0])
        #filename
        fn = pieces[1]
        #index data.  v is only the score value for the index data
        v = [float(x.strip().split(":")[-1]) for x in pieces[2:]]
        #throw an error if the size of v does not match the size of the coeffs
        assert len(v) == len(coeffs)
        #calculate the dumbscore by multiplying the coeff value by the v value and
        #taking the sum of everything
        dumbscore = sum([cc * vv for cc,vv in zip(coeffs,v)])

        if dumbscore >= thresh: guess = 1
        else: guess = 0

        accuracy = ""
        mark = ""
        if cl in [0,2]:
            mark = 1
            if guess == 1: accuracy = 1
            else: accuracy = 0
        elif cl in [1,3]:
            mark = 0
            if guess == 0: accuracy = 1
            else: accuracy = 0

        #only do analysis on test data
        if cl in [2,3]: output_dict[fn] = {"score":dumbscore,"class":cl,"guess":guess,"thresh":thresh,"accuracy":accuracy,"mark":mark}

    return output_dict

def calculate_redundancy(active_rois):
	#import transcripts and export the appropriate redundancy value
	#we are going to make an sum list and a mark list
	
    super_dict={}
    for roi in active_rois:
        super_dict[roi]=transcript_to_dict(roi)

    #generate a list of filenames using a dict
    filename_dict={}
    for roi_key in super_dict.keys():
        for filename_key in super_dict[roi_key].keys():
            filename_dict[filename_key]=""

    filename_list=filename_dict.keys()

    data_list=[]
    mark_list=[]
    for file in filename_list:
        sum=0
        mark_mini_list=[]
        for roi_key in super_dict.keys():
            if file in super_dict[roi_key].keys():
                sum+=super_dict[roi_key][file]["guess"]
                mark_mini_list.append(super_dict[roi_key][file]["mark"])
            else:
                mark_mini_list=[0,1]
		#all marks must be identical.
        if not fun.checkEqual(mark_mini_list): continue
        data_list.append(sum)
        mark_list.append(mark_mini_list[0])
	
    thresh,J_abs,J,sen,spec=fun.threshold_finder(data_list,mark_list)

    return thresh,J,sen,spec

def serial_test(serial_map,use_all_rois=True):
    #make a new serial map setup as [serial_number:[old_roi,new_roi]]
    #determine what the max serial number is
    max_sn=0
    for key in serial_map.keys():
        max_sn=max(serial_map[key][0],max_sn)

    new_serial_map={}
    for sn in range(1,max_sn+1):
        new_serial_map[sn]=[]

    for key in serial_map.keys():
        new_serial_map[serial_map[key][0]].append(key)

    #print new_serial_map

    minimum_count=2
    best_J=0
    best_sen=0
    best_spec=0
    best_thresh=0
    best_rois=[]

    for serial_number in new_serial_map.keys():
        available_ROIs=new_serial_map[serial_number]
        available_ROIs_count=len(new_serial_map[serial_number])
        if use_all_rois:
            ROI_count_loop=[available_ROIs_count]
        else:
            ROI_count_loop=range(minimum_count,available_ROIs_count+1)
        for ROI_count in ROI_count_loop:
            if use_all_rois:
                active_ROIs_loop=[available_ROIs]
            else:
                active_ROIs_loop=list(it.combinations(available_ROIs,ROI_count))
            for active_ROIs in active_ROIs_loop:
                thresh,J,sen,spec=calculate_redundancy(active_ROIs)
                #print [thresh,J,sen,spec,active_ROIs]
                if J>best_J:
                    best_J=J
                    best_sen=sen
                    best_spec=spec
                    best_thresh=thresh
                    best_rois=active_ROIs
            
        print [serial_number,best_thresh,best_J,best_sen,best_spec,best_rois]

def model_header_compile(active_rois):
    #make big .c files
    output_file_list=[]
    for active_roi in active_rois:
        os.system("python make_model_header.py "+str(int(active_roi))) #make model files for each ROI
        output_file_list.append("model_data_"+str(int(active_roi))+".c")
        print "finished", active_roi
    with open('output_file.txt','wb') as wfd:
        for f in output_file_list:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd, 1024*1024*10)
    os.rename('output_file.txt', 'model_data_fin.c')