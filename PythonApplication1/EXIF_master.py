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
directory='C:\Users\gttho\Documents\Visual Studio 2017\Projects\PythonApplication1\PythonApplication1\Resources\FigureImport'
#directory='G://Google Drive//Original Images//Embroidery//171122 TriPhoto Test//photos//AllPhotos'
directory="G:\Google Drive\Original Images\Embroidery\_Complete Photo Sets\Mi.50_v1app"
directory="G://Google Drive//Original Images//Embroidery//171206 WB_Exp//Auto6"
directory="G://Google Drive//Original Images//Embroidery//171206 WB_Exp//Locked1.3"
#directory="G://Google Drive//Original Images//Embroidery//171206 ExpWB Test"
directory="G://Google Drive//Original Images//Embroidery//171207 ExpWB Test2"
directory="G://Google Drive//Original Images//Embroidery//171208 ExpWb NewApproach"
directory="G://Google Drive//Original Images//Embroidery//171210 New_Exp_Wb_BigPhotoset"
directory="G://Google Drive//Original Images//Embroidery//171213 New_Exp_Wb_AdderSet"
directory_print1="G://Google Drive//Original Images//Embroidery//test_destination//print_103"
directory_print2="G://Google Drive//Original Images//Embroidery//test_destination//print_109"
directory_blank="G://Google Drive//Original Images//Embroidery//test_destination//blank"
directory_move="G://Google Drive//Original Images//CanadaGoose//171214 PhotoDump_PicsIWant"
directory="G://Google Drive//Original Images//CanadaGoose//171214 PhotoDump2"
directory="G://Google Drive//Original Images//CanadaGoose//cheeky little temp folder"
directory="G://Google Drive//Color_regression//images//temp_photo_fixer"
directory="G://Temp_EXIF_fixer"
#directory="G://Google Drive//Color_regression//images//180111 Campus_vs_LightTent//LightTent"
pat, dirs, files = os.walk(directory).next()
numfil=len(files)
total_counter=22*numfil #there are 22 colorspaces

##############__________PREP HOLDING ARRAYS_______############3
coeffs=np.zeros((total_counter,5))
date='dummy'
counter1=0 #use for the total counter, to input values into the coeffs array
for filename in os.listdir(directory):
    #if (filename[-4:].lower()=='.jpg') and (filename[:9]=='PE_Legacy'):
    print filename
    if (filename[-4:].lower()=='.jpg'):
        try:
        #if True:
            #grab EXIF data
            filname=os.path.join(directory, filename)
            exif_dict=piexif.load(filname)
            user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
            user_comment_prime=copy.copy(user_comment)

            ############CONVERT TEXT A TO TEXT B##########
            #convert Count value
            #user_comment=string.replace(user_comment,'Count:1','Count:20')
            #user_comment=string.replace(user_comment,'Count:0','Count:30')
            #user_comment=string.replace(user_comment,'Count:-1','Count:10')
            #user_comment=string.replace(user_comment,'Count:200','Count:10')
            user_comment=string.replace(user_comment,'PhotoSet:103','PhotoSet:104')

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
            #user_comment+=',ExpBias:-6,WhiteBalanceLocked:true'
            #print user_comment

            ###########MOVE BASED ON EXIF###########
            #print_string2_2="PhotoSet:110"
            #print_string2_1="PhotoSet:109"
            #blank_string1="PhotoSet:102"
            #blank_string2="PhotoSet:101"
            #print_string1_2="PhotoSet:103"
            #print_string1_1="PhotoSet:104"

            #if blank_string1 in user_comment:
            #    shutil.move(os.path.join(directory,filename),os.path.join(directory_blank,filename))
            #    print "blank1 mover"
            #if blank_string2 in user_comment:
            #    shutil.move(os.path.join(directory,filename),os.path.join(directory_blank,filename))
            #    print "blank2 mover"
            #if print_string1_1 in user_comment:
            #    shutil.move(os.path.join(directory,filename),os.path.join(directory_print1,filename))
            #    print "print1 mover"
            #if print_string1_2 in user_comment:
            #    shutil.move(os.path.join(directory,filename),os.path.join(directory_print1,filename))
            #    print "print1 mover"    
            #if print_string2_1 in user_comment:
            #    shutil.move(os.path.join(directory,filename),os.path.join(directory_print2,filename))
            #    print "print2 mover"
            #if print_string2_2 in user_comment:
            #    shutil.move(os.path.join(directory,filename),os.path.join(directory_print2,filename))
            #    print "print2 mover"    

            #move_string1="Ink:50,Binder:10,Solvent:20,Formulation:BR Formulation"
            #move_string2="Ink:1.25,Binder:73,Solvent:25.75,Formulation:S.FM1 - 50/50-B - PM"
            #move_string1="XXXXXXXXXXXXXXXXXXXXX"
            #move_string2="XXXXXXXXXXXXXXXXXXXXXXXXXX"
            #move_string3="Blank"
            #if (move_string1 in user_comment) or (move_string2 in user_comment) or (move_string3 in user_comment):
            #    shutil.move(os.path.join(directory,filename),os.path.join(directory_move,filename))
            #    print "MOVE"   
                
            ##############OVERWRITE ALL USER DATA#############
            #print user_comment
            ##user_comment='Ink:0,Binder:0,Solvent:0,Formulation:Blank,Modified:M0'
            #user_comment='Ink:50,Binder:10,Solvent:10,Formulation:BR Formulation,Modified:CG'

            ###############RENAME FILE BASED ON EXIF############
            #ps_list=['101','102','103','104','109','110']
            #append_value=''
            #for ps_string in ps_list:
            #    ps_checker="PhotoSet:"+ps_string
            #    if ps_checker in user_comment:
            #        append_value='ps'+ps_string
            #new_filename=filname[:-4]+'_'+append_value+'.jpg'
            #os.rename(filname,new_filename)

            #print user_comment
            print counter1

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