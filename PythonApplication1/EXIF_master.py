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
secret_stuff=3005
#file=raw_input('Filename with extension: ')

##########_________USER INPUT___________###########
print ""
directory_start=raw_input("Enter starting directory: ")
all_file_switch=bool(int(raw_input("Process all files in subfolders? 1/0 ")))

#Define exif test
print "Time to pick what EXIF modifications you need.  The options are as follows and changes are applied in this order."
print "0 - Search for text in EXIF"
print "1 - Convert Text 'A' to 'B'"
print "2 - Convert Text 'A' to 'B' with a condition"
print "3 - Add text directly"
print "4 - Move file based on UserComments"
print "5 - Move file based on File Name"
print "6 - Change UserComments based on File Path"
print "7 - Overwrite ALL UserComments"
print "8 - Rename file based on UserComments"
print ""

number_list_raw=raw_input("Enter the modifications you'd like separated by spaces: ")
test_type=map(int,number_list_raw.split())

##Input for specific test_types
serach_for_text_in_string=""
if 0 in test_type:
    print "Input for 0 - Search for text in EXIF"
    serach_for_text_in_string=raw_input("Enter the text you will search for: ")
if 1 in test_type:
    print "Input for 1 - Convert Text 'A' to 'B'"
    convert_start_text=raw_input("Input the text you'll look for in the UserComments: ")
    convert_end_text=raw_input("Input the text you'll replace the above input with: ")
if 2 in test_type:
    print "Input for 2 - Convert Text 'A' to 'B' with a condition"
    convert_condition_text=raw_input("Input the text that, if found, will trigger the conversion: ")
    convert_condition_start_text=raw_input("Input the text you'll look for in the UserComments: ")
    convert_condition_end_text=raw_input("Input the text you'll replace the above input with: ")
if 3 in test_type:
    print "Input for 3 - Add text directly"
    direct_add_to_beginning=bool(int(raw_input("Would you like to add the text to the beginning of UserComment?  If not, it'll be added to the end. 1/0 ")))
    direct_add_text=raw_input("Input the text you will add to the UserComment.  Don't forget commas! : ")
if 4 in test_type:
    if all_file_switch:
        print "I'm sorry.  Files cannot be moved if all files in subfolders are being scanned."
        test_type[:] = [x for x in test_type if x != 4]  
    else:
        print "Input for 4 - Move file based on UserComments"
        uc_move_list_string=raw_input("Input the text that will trigger the move, separated by commas: " )
        uc_move_list=uc_move_list_string.split(",")
        uc_directory_move=raw_input("Input the directory that the files will be moved to: ")
if 5 in test_type:
    if all_file_switch:
        print "I'm sorry.  Files cannot be moved if all files in subfolders are being scanned."
        test_type[:] = [x for x in test_type if x != 5] 
    else:
        print "Input for 5 - Move file based on File Name"
        fn_move_list_string=raw_input("Input the text that will trigger the move, separated by commas: " )
        fn_move_list=fn_move_list_string.split(",")
        fn_directory_move=raw_input("Input the directory that the files will be moved to: ")
if 6 in test_type:
    print "Input for 6 - Change UserComments based on File Path"
    fp_exif_list_string=raw_input("Input the keys you care about, from the bottom folder to the top folder.  Separate by commas. ")
    fp_exif_list=fp_exif_list_string.split(",")
    fp_overwrite=bool(int(raw_input("Would you like to overwrite the existing UserComments? ")))
    if fp_overwrite:
        password_attempt=int(raw_input("Enter the password to allow for the full overwrite of UserComments: "))
        if password_attempt==secret_stuff:
            print "Password is correct.  Current UserComments will be deleted."
        else:
            print "Password is incorrect.  Additional data will be appended to start or end of existing comments."
            fp_overwrite=False
    if not fp_overwrite:
        fp_add_to_start=bool(int(raw_input("Would you like to add the new UserComments info to the start? If not, it will be added to the end 1/0 ")))
if 7 in test_type:
    print "Input for 7 - Overwrite ALL UserComments"
    password_attempt=int(raw_input("Enter the password to allow for the full overwrite of UserComments: "))
    if password_attempt==secret_stuff:
        print "Password is correct.  Current UserComments will be deleted."
        overwrite_text=raw_input("Input text that you will replace the UserComments with: ")
    else:
        print "Password is incorrect.  Step index '7' will be skipped"
        test_type[:] = [x for x in test_type if x != 7]    
if 8 in test_type:
    print "Input for 8 - Rename file based on UserComments"
    rename_list_string=raw_input("Input the keys that, if found, will be appended to the filename.  Separate with commas: ")
    rename_list=rename_list_string.split(",")
    nf_add_to_start=bool(int(raw_input("Would you like the additional filename to go to the start?  If not, it will be appended to the end. 1/0 ")))

##Recap the inputs the user gave, and force final confirmation
print ""
print ""
print "Let's review..."
if 0 in test_type:
    print "I'll look for: "+serach_for_text_in_string
if 1 in test_type:
    print "I'll convert "+convert_start_text+" to "+convert_end_text
if 2 in test_type:
    print "I'll convert "+convert_condition_start_text+" to " +convert_condition_end_text +" only if the UserComments contains "+convert_condition_text
if 3 in test_type:
    print "I'll add the text "+direct_add_text+" to the ",
    if (direct_add_to_beginning):
        print "start",
    else:
        print "end",
    print " of UserComments"
if 4 in test_type:
    print "I'll be moving files to the directory: "+uc_directory_move
    print "I'll only do that though if the UserComments contains any of: ",
    for uc_move_list_element in uc_move_list:
        print uc_move_list_element,
        print ",",
    print ""
if 5 in test_type:
    print "I'll be moving a file to the directory: "+fn_directory_move
    print "I'll only do that though if the filename contains any of: ",
    for fn_move_list_element in fn_move_list:
        print fn_move_list_element,
        print ",",
    print ""
if 6 in test_type:
    if fp_overwrite:
        print "I'll be replacing ",
    elif fp_add_to_start:
        print "I'll be addinng to the start of ",
    else:
        print "I'll be adding to the end of ",
    print "the UserComments based on the tree structure of the file path"
    print "The list of keys I'll be using are: ",
    for single_key in fp_exif_list:
        print single_key,
        print ","
    print ""
    if fp_overwrite:
        print "***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***"
        print "I'M GOING TO BE DELETING THE EXISTING USERCOMMENTS. THIS CANNOT BE UNDONE."
if 7 in test_type:
    print "I'll be replacing the current UserComments with "+overwrite_text
    print "***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***WARNING***"
    print "I'M GOING TO BE DELETING THE EXISTING USERCOMMENTS. THIS CANNOT BE UNDONE."
if 8 in test_type:
    print "I'll be modifying the filename.  I'll add the following to it if they are found in the UserComments: "
    for rename_list_element in rename_list:
        print rename_list_element,
        print ","
    print "The additional filename elements will be added to the ",
    if nf_add_to_start:
        print "start",
    else:
        print "end",
    print " of the filename."

print ""
continue_switch=bool(int(raw_input("Is everything correct?  This is your last chance to make changes. 1/0 ")))
if not continue_switch:
    sys.exit("UserComment Modificaiton Terminated")

#############3_______PHOTO IMPORT_______##############

#directory_move="C://Users//gttho//Resilio Sync//Developer//PhotoExports//180330 CG Refined Conc Sweep Test - BadPics//50mS"

#directory_start="C://Users//gttho//Documents//Visual Studio 2017//Projects//PythonApplication1//PythonApplication1//bad_pics"
#directory_start="D://Google Drive//RM Stress Testing//Print_lowlux"
#directory_start="D://Offline//AryllaTemp"
#adder_string=["50mS","75mS","100mS"]
adder_string=[""]
for adder in adder_string:
    #all_file_switch=0

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
                if 0 in test_type:
                    search_string=serach_for_text_in_string 
                    if search_string in user_comment:
                        finder_counter+=1
                        print filename,
                        print "  FOUND:  ",
                        print search_string
                    print "Finder counter: ",
                    print finder_counter

                ############CONVERT TEXT A TO TEXT B ##########
                #convert Count value
                if 1 in test_type:
                    user_comment=string.replace(user_comment,convert_start_text,convert_end_text)

                ############CONVERT TEXT A TO TEXT B with switch ##########
                if 2 in test_type:
                    switch_string=convert_condition_text
                    if switch_string in user_comment:
                        user_comment=string.replace(user_comment,convert_condition_start_text,convert_condition_end_text)

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
                if 3 in test_type:
                    if direct_add_to_beginning:
                        user_comment=direct_add_text+user_comment
                    else:
                        user_comment=user_comment_direct_add_text
                    print user_comment
                    

               
                ###########MOVE BASED ON EXIF###########
                if 4 in test_type:
                    #uc_lower=user_comment.lower()
                    uc_lower=user_comment

                 #   move_list=["mark:1,project:cg,wash no:0,dry clean no:0,sample no:1,conc:40,test type:wash,","mark:1,project:cg,wash no:1,dry clean no:0,sample no:1,conc:40,test type:wash,","mark:1,project:cg,wash no:2,dry clean no:0,sample no:1,conc:40,test type:wash,","mark:1,project:cg,wash no:3,dry clean no:0,sample no:1,conc:40,test type:wash,","mark:1,project:cg,wash no:4,dry clean no:0,sample no:1,conc:40,test type:wash,",",app:243,mark:1,project:cg,wash no:4,dry clean no:0,conc:40,test type:wash,sample no:1,"]
                    move_list=uc_move_list

                    for move_string in move_list:
                        if move_string in uc_lower:
                            shutil.move(os.path.join(directory,filename),os.path.join(uc_directory_move,filename))
                            print "moved"
                


                ###########MOVE BASED ON FILE NAME###########
                if 5 in test_type:
                    filename_lower=filename.lower()
                    #uc_lower=user_comment

                    move_list=fn_move_list

                    for move_string in move_list:
                        if move_string in filename_lower:
                            shutil.move(os.path.join(directory,filename),os.path.join(fn_directory_move,filename))
                            print "moved"


                ###########CHANGE EXIF BASED ON FILE PATH############
                if 6 in test_type:
                    filname_mod=filname.replace('//',"////")
                    filname_mod=filname.replace('\\',"////")
                    filname_list=filname_mod.split("////")
                    print filname_mod
                    print filname_list
                    del filname_list[-1]  #Deletes the filename from the list
                    filname_list.reverse() #reverses the order so that the suff you care about is in the beginning

                    key_list=fp_exif_list
                    print key_list
                    #key_list=['Location']
                    uc_new=','
                    for key_iter in range(len(key_list)):
                        uc_new+=key_list[key_iter]+":"+filname_list[key_iter]+","

                    overwrite_switch=fp_overwrite
                    if overwrite_switch:
                        user_comment=uc_new
                    elif fp_add_to_start:
                        user_comment=uc_new+user_comment
                    else:
                        user_comment=user_comment+uc_new



                ##############OVERWRITE ALL USER DATA#############
                if 7 in test_type:
                    print user_comment
                    user_comment=overwrite_text
                   # user_comment='Ink:0,Binder:'+str(t_diff_hours)+',Solvent:'+str(int(round(t_diff_hours)))+',Formulation:Ink,Modified:'+adder

                ###############RENAME FILE BASED ON EXIF############
                if 8 in test_type:
                    filname_mod=filname.replace('//',"////")
                    filname_mod=filname_mod.replace('\\',"////")
                    filname_list=filname_mod.split("////")

                    old_filename=filname_list[-1]
                    old_filename=old_filename.replace(".jpg","")

                    del filname_list[-1]  #Deletes the filename from the list

                    ps_list=rename_list
            
                    append_value=''
                    #print user_comment
                    rename_file=False
                    for ps_string in ps_list:
                        ps_checker=ps_string
                        if ps_checker in user_comment:
                            rename_file=True
                            append_value+=ps_string
            
                    #ps_list=['Binder:2','Binder:1']

                    #print user_comment
                    #for ps_string in ps_list:
                    #    ps_checker=ps_string
                    #    if ps_checker in user_comment:
                    #        append_value+='_'+ps_string[:-1]
            
                    #date=exif_dict['Exif'][36867]
                    append_value=append_value.replace(":",'')
                    append_value=append_value.replace("\\",'')
                    append_value=append_value.replace("/",'')
                    #append_value=date
                   
                    #uc_noColon=user_comment.replace(":","-")
                    #uc_noColon=uc_noColon.replace("/","")
                    #append_value=uc_noColon

                    if nf_add_to_start:
                        new_filename=append_value+"_"+old_filename+".jpg"
                    else:
                        new_filename=old_filename+"_"+append_value+".jpg"

                    new_filename_path=""
                    for file_bit in filname_list:
                        new_filename_path=new_filename_path+file_bit+"\\"


                    new_filename=new_filename_path+new_filename
                    #new_filename=filname[:-4]+'_'+append_value+'.jpg'
                    #new_filename=filname[:-12]+'_'+append_value+'_'+filname[-12:-4]+'.jpg'

                    if rename_file:
                        print "Renamed file to "+new_filename
                        os.rename(filname,new_filename)
                        filname=copy.copy(new_filename)

               # print user_comment
               # print counter1
                

                #Only modify EXIF if a change has been applied
                if not user_comment==user_comment_prime:
                    user_comment = piexif.helper.UserComment.dump(user_comment)
                    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes,filname)
                    counter1+=1
                    print "New Exif: "+user_comment,
                    print "; A total of "+str(counter1)+" files have been modified"
            #except:
            #    print "************************************skipper************************************"
            #    null=1