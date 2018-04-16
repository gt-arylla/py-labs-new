#Identifies the best colorspaces and averages them together
#####################__________LIBRARIES______________################3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as pltfig
#import qrtools
import os
import winsound
import sys
import time

#############____________CONSTANTS_______________##########33
resize=0.5
b_ratio=0.1105
resize_thresh=130000


c_thresh=.0
stdev_t=110
avg_thresh=60

eq_hist='n'
clay='n'
s3='n'
r_check='y'

#if user gave a path in the command line, use it.  Otherwise, use default
if len(sys.argv)>1:
	directory=sys.argv[1]
	directory=directory.replace("/","\\")
else:
	directory='\FigureImport'


print directory

tight_range=[135,175]
#default values are yes, no.  If user gives a third console input, use default values
if len(sys.argv)>2:
	thresh='n'
	eq_hist='n'
	clay='n'
	s3='n'
	pic_show='y'
else:
	pic_show=raw_input('Show Pic? y/n ')
	start_value=int(raw_input('Start value? (int) '))
	end_value=int(raw_input('End value? (int) '))
	tight_range=[start_value,end_value]
	#file=raw_input('Filename with extension: ')

#############3_______PHOTO IMPORT_______##############
#directory='G:\Developer\py-labs\FigureImport'
pat, dirs, files = os.walk(directory).next()
numfil=len(files)
total_counter=22*numfil #there are 22 colorspaces

##############__________PREP HOLDING ARRAYS_______############3
coeffs=np.zeros((total_counter,5))


counter1=0 #use for the total counter, to input values into the coeffs array
for filename in os.listdir(directory):
	if filename[-4:].lower()=='.jpg':
		b_ratio=0.1105
		filname=os.path.join(directory, filename)
		# print filname
		#############_____PREP PHOTO__________###############3
		img=cv2.imread(filname)
		# print img
		# print img.shape
		# print img.size
		if img.size>resize_thresh and r_check=='y':
			#resize=np.sqrt(resize_thresh/float(img.size))
			# print resize
			img=cv2.resize(img,None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
			#b_ratio=b_ratio*resize
			# print b_ratio
			
		# pltfig.Figure(figsize=[12,10], dpi=200, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
		# plt.subplot(461)
		# i_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		# plt.imshow(i_rgb)
		# plt.title(filename,fontsize=8)
		# plt.axis('off')

		###########__Set up Threshold_____##############
		i_quick=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		box=int(round(min(i_quick.shape)*b_ratio))
		if box%2==0: #forces the box to be an odd number
			box=box+1
		# print box
		############___________PREP EXTRACTIONS_______############
		m_1=np.array([1,0,0]).reshape((1,3))
		m_2=np.array([0,1,0]).reshape((1,3))
		m_3=np.array([0,0,1]).reshape((1,3))
		
		
		counter0=0 #use for the winner counter, to input images into 'winner_figs' and 'winner_coeffs'
		pic_use=0 #used to input whether or not the picture was used into the coeff array
		############________COLORSPACES_______#############
		i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
		i_ext=cv2.transform(i_ext,m_3)
		i_ext=np.bitwise_not(i_ext)
		name='HSV H'
		i_ext_temp=cv2.equalizeHist(i_ext)
		cv2.imwrite("export.jpg",i_ext_temp)
		cv2.imwrite("basis.jpg",img);
		plt.figure(5)
		plt.imshow(i_ext,clim=(tight_range[0],tight_range[1]),cmap="jet")
		plt.axis('off')
		if pic_show=="y":
			plt.show()
		else:
			plt.savefig(filname+".png")
		
		# plt.figure(6)
		# plt.imshow(i_ext,clim=(tight_range[0],tight_range[1]),cmap="jet")
		# plt.show()


		counter1=counter1+1
		# print counter1/float(numfil)
