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
import piexif
import piexif.helper
import string
import copy
import shutil

#############____________CONSTANTS_______________##########33

user_comment='test_comment'
#file=raw_input('Filename with extension: ')

#############3_______PHOTO IMPORT_______##############

directory_move="G://Google Drive//Original Images//Pumped Kicks//180207 Algo_Test//Blank"

directory_start="G://Google Drive//Original Images//Pumped Kicks//180207 Algo_Test//"
adder_string=["50mS","75mS","100mS"]
#adder_string=[""]
for adder in adder_string:
    directory=directory_start+adder
    pat, dirs, files = os.walk(directory).next()
    numfil=len(files)
    total_counter=22*numfil #there are 22 colorspaces

    ##############__________PREP HOLDING ARRAYS_______############3
    coeffs=np.zeros((total_counter,5))
    date='dummy'
    counter1=0 #use for the total counter, to input values into the coeffs array
    finder_counter=0
    for filename in os.listdir(directory):
        #if (filename[-4:].lower()=='.jpg') and (filename[:9]=='PE_Legacy'):
        #print filename
    
        if (filename[-4:].lower()=='.jpg'):
            try:
            #if True:
                #grab EXIF data
                filname=os.path.join(directory, filename)
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
                search_string="Solvent:c"
                if search_string in user_comment:
                    finder_counter+=1
                    print filename,
                    print "  FOUND:  ",
                    print search_string

                ############CONVERT TEXT A TO TEXT B ##########
                #convert Count value
                if (0):
                    user_comment=string.replace(user_comment,'Dtamp','Stamp')

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
                    user_comment+=',Ink:0,Binder:2,Solvent:0,Formulation:Ink,Mod:bsb'
                    print user_comment

                ###########MOVE BASED ON EXIF###########
                if (0):
                    #uc_lower=user_comment.lower()
                    uc_lower=user_comment

                    move_list=["Formulation:Blank"]
                    #move_string2="Binder:2"

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

                ##############OVERWRITE ALL USER DATA#############
                if(0):
                    print user_comment
                    #user_comment='Ink:0,Binder:0,Solvent:0,Formulation:Blank,Modified:M0'
                    user_comment='Ink:0,Binder:'+str(t_diff_hours)+',Solvent:'+str(int(round(t_diff_hours)))+',Formulation:Ink,Modified:'+adder

                ###############RENAME FILE BASED ON EXIF############
                if(1):
                    #ps_list=['Ink','Blank']
            
                    #append_value=''
                    #print user_comment
                    #for ps_string in ps_list:
                    #    ps_checker=ps_string
                    #    if ps_checker in user_comment:
                    #        append_value+=ps_string
            
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
                   
                    uc_noColon=user_comment.replace(":","-")
                    append_value=uc_noColon

                    new_filename=filname[:-4]+'_'+append_value+'.jpg'
                    #new_filename=filname[:-12]+'_'+append_value+'_'+filname[-12:-4]+'.jpg'
                    os.rename(filname,new_filename)

                #print user_comment
                print counter1
                print "Finder counter: ",
                print finder_counter

                #Only modify EXIF if a change has been applied
                if not user_comment==user_comment_prime:
                    user_comment = piexif.helper.UserComment.dump(user_comment)
                    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes,filname)
                    counter1+=1
            except:
                print "************************************skipper************************************"
                null=1