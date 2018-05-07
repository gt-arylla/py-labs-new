#Identifies the best colorspaces and averages them together
#####################__________LIBRARIES______________################3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as pltfig
import os
import winsound
import sys
from glob import glob
import time
import piexif
import piexif.helper
import string
import copy
import shutil

#############____________CONSTANTS_______________##########33

user_comment='test_comment'
#file=raw_input('Filename with extension: ')

#############3_______PHOTO IMPORT_______##############

directory_move="C://Users//gttho//Resilio Sync//Developer//PhotoExports//180330 CG Refined Conc Sweep Test - BadPics//50mS"

directory_start="C://Users//gttho//Documents//Visual Studio 2017//Projects//PythonApplication1//PythonApplication1//bad_pics"
directory_start="D://Google Drive//RM Stress Testing//Print_lowlux"
#directory_start="D://Offline//AryllaTemp"
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
    
        if (filename[-4:].lower()=='.jpg'):
            #try:
            if True:
                #grab EXIF data
                if all_file_switch:
                    filname=filename
                else:
                    filname=os.path.join(directory, filename)
                print filname
                exif_dict=piexif.load(filname)
                user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
                user_comment_prime=copy.copy(user_comment)

                #grab date as YYYYMMDDhhmmss
                date=exif_dict['Exif'][36867]
                date=date.replace(":",'')
                date=date.replace(" ",'')
                year=int(date[0:4])
                month=int(date[4:6])
                day=int(date[6:8])
                hour=int(date[8:10])
                minute=int(date[10:12])
                second=int(date[12:14])

                #### Calculate time since t0 ####
                if(0):
                    t0="2018-02-04 17:10:00"
                    t1_raw=exif_dict['Exif'][36867]
                    t1_date=t1_raw[0:10]
                    t1_date=t1_date.replace(":","-")
                    t1=t1_date+t1_raw[10:]
                    t_diff_np=np.datetime64(t1) - np.datetime64(t0)
                    t_diff_seconds=t_diff_np.item().total_seconds()
                    t_diff_hours=t_diff_seconds/3600.0


                ############ Search for text in EXIF ############
                if(0):
                    search_string="mark:print"
                    if search_string in user_comment:
                        finder_counter+=1
                        print filename,
                        print "  FOUND:  ",
                        print search_string
                    print "Finder counter: ",
                    print finder_counter

                ############CONVERT TEXT A TO TEXT B ##########
                #convert Count value
                if (1):
                    user_comment=string.replace(user_comment,'mark:0,','mark:1,')
                    #user_comment=string.replace(user_comment,'mark:blank,','mark:0,')

                ############CONVERT TEXT A TO TEXT B with switch ##########
                if (0):
                    switch_string="cap:cap 4"
                    if switch_string in user_comment:
                        user_comment=string.replace(user_comment,'mark:0','mark:1')
                    #user_comment=string.replace(user_comment,'mark:blank,','mark:0,')

                ############Fancy Convert (date, commas, convert to number)#############
                #if counter1%3==0:
                #    #pull new date value
                #    date=exif_dict['Exif'][36867]
                #    date=date.replace(":",'')
                #    date=date.replace(" ",'')

                ##add date to exif data as bunch
                #user_comment+=',PhotoBunch:'+date

                ##fix absence of comma
                #if not ",ExpBias" in user_comment:
                #    user_comment=string.replace(user_comment,'ExpBias',',ExpBias')

                #Convert 'true' and 'false' to '1' and '0'
                #if "WhiteBalanceLocked:true" in user_comment:
                #    user_comment=string.replace(user_comment,'WhiteBalanceLocked:true','WhiteBalanceLocked:1')
                #if "WhiteBalanceLocked:false" in user_comment:
                #    user_comment=string.replace(user_comment,'WhiteBalanceLocked:false','WhiteBalanceLocked:0')

                ###########Direct Add#################
                if(0):
                    user_comment=user_comment+"location:labbench,"
                    print user_comment
                    

               
                ###########MOVE BASED ON EXIF###########
                if (0):
                    #uc_lower=user_comment.lower()
                    uc_lower=user_comment

                 #   move_list=["mark:1,project:cg,wash no:0,dry clean no:0,sample no:1,conc:40,test type:wash,","mark:1,project:cg,wash no:1,dry clean no:0,sample no:1,conc:40,test type:wash,","mark:1,project:cg,wash no:2,dry clean no:0,sample no:1,conc:40,test type:wash,","mark:1,project:cg,wash no:3,dry clean no:0,sample no:1,conc:40,test type:wash,","mark:1,project:cg,wash no:4,dry clean no:0,sample no:1,conc:40,test type:wash,",",app:243,mark:1,project:cg,wash no:4,dry clean no:0,conc:40,test type:wash,sample no:1,"]
                    move_list=["conc:50"]

                    for move_string in move_list:
                        if move_string in uc_lower:
                            shutil.move(os.path.join(directory,filename),os.path.join(directory_move,filename))
                            print "moved"
                


                ###########MOVE BASED ON FILE NAME###########
                if (0):
                    filename_lower=filename.lower()
                    #uc_lower=user_comment

                    move_list=["("]

                    for move_string in move_list:
                        if move_string in filename_lower:
                            shutil.move(os.path.join(directory,filename),os.path.join(directory_move,filename))
                            print "moved"


                ###########CHANGE EXIF BASED ON FILE PATH############
                if(0):
                    filname_mod=filname.replace('//',"////")
                    filname_list=filname_mod.split("////")
                    del filname_list[-1]  #Deletes the filename from the list
                    filname_list.reverse() #reverses the order so that the suff you care about is in the beginning

                    key_list=['Bow','Angle','Background','Distance','Flash']

                    #key_list=['Location']
                    uc_new=''
                    for key_iter in range(len(key_list)):
                        uc_new+=key_list[key_iter]+":"+filname_list[key_iter]+","

                    overwrite_switch=1
                    if overwrite_switch:
                        user_comment=uc_new
                    else:
                        user_comment=uc_new+user_comment



                ##############OVERWRITE ALL USER DATA#############
                if(0):
                    print user_comment
                    user_comment='AP:AP0,Exposure Bias:EV0,Size:0.50 inch,Location:LIH,Colour:Black,Brand:Jordans,Formulation:Blank,LUX:l100,'
                   # user_comment='Ink:0,Binder:'+str(t_diff_hours)+',Solvent:'+str(int(round(t_diff_hours)))+',Formulation:Ink,Modified:'+adder

                ###############RENAME FILE BASED ON EXIF############
                if(0):
                    ps_list=['Print','Blank']
            
                    append_value=''
                    print user_comment
                    for ps_string in ps_list:
                        ps_checker=ps_string
                        if ps_checker in user_comment:
                            append_value+=ps_string
            
                    #ps_list=['Binder:2','Binder:1']

                    #print user_comment
                    #for ps_string in ps_list:
                    #    ps_checker=ps_string
                    #    if ps_checker in user_comment:
                    #        append_value+='_'+ps_string[:-1]
            
                    #date=exif_dict['Exif'][36867]
                    #date=date.replace(":",'')
                    #date=date.replace(" ",'')
                    #append_value=date
                   
                    #uc_noColon=user_comment.replace(":","-")
                    #uc_noColon=uc_noColon.replace("/","")
                    #append_value=uc_noColon

                    new_filename=filname[:-4]+'_'+append_value+'.jpg'
                    #new_filename=filname[:-12]+'_'+append_value+'_'+filname[-12:-4]+'.jpg'
                    print new_filename
                    os.rename(filname,new_filename)

               # print user_comment
                print counter1
                

                #Only modify EXIF if a change has been applied
                if not user_comment==user_comment_prime:
                    user_comment = piexif.helper.UserComment.dump(user_comment)
                    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes,filname)
                    counter1+=1
                    print "New Exif: "+user_comment
            #except:
            #    print "************************************skipper************************************"
            #    null=1