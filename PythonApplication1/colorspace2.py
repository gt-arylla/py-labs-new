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
resize_thresh=1300000



c_thresh=.0
stdev_t=110
avg_thresh=60

#if user gave a path in the command line, use it.  Otherwise, use default
if len(sys.argv)>1:
	directory=sys.argv[1]
	directory=directory.replace("/","\\")
else:
	directory='FigureImport//'
#directory="G://Google Drive//Original Images//Embroidery//171130 TriPhotoTest MaskingSubset//"

print directory

#default values are yes, no.  If user gives a third console input, use default values
if len(sys.argv)>2:
	thresh='n'
	pic_show='y'
else:
	thresh=raw_input('Apply thresholds? y/n ')
	pic_show=raw_input('Show Pic? y/n ')
#file=raw_input('Filename with extension: ')

#############3_______PHOTO IMPORT_______##############
pat, dirs, files = os.walk(directory).next()
numfil=len(files)
total_counter=22*numfil #there are 22 colorspaces

##############__________PREP HOLDING ARRAYS_______############3
coeffs=np.zeros((total_counter,5))


counter1=0 #use for the total counter, to input values into the coeffs array
for filename in os.listdir(directory):
	b_ratio=0.1105
	if filename[-4:].lower()=='.jpg' or filename[-4:].lower()=='.png':
		filname=os.path.join(directory, filename)
		#############_____PREP PHOTO__________###############3
		img=cv2.imread(filname)
		if img.size>resize_thresh:
			img=cv2.resize(img,None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
			#b_ratio=b_ratio*resize
		pltfig.Figure(figsize=[120,100], dpi=200, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
		plt.subplot(461)
		i_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		plt.imshow(i_rgb)
		plt.title(filename,fontsize=8)
		plt.axis('off')

		###########__Set up Threshold_____##############
		i_quick=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		box=int(round(min(i_quick.shape)*b_ratio))
		if box%2==0: #forces the box to be an odd number
			box=box+1

		############___________PREP EXTRACTIONS_______############
		m_1=np.array([1,0,0]).reshape((1,3))
		m_2=np.array([0,1,0]).reshape((1,3))
		m_3=np.array([0,0,1]).reshape((1,3))
		
		counter0=0 #use for the winner counter, to input images into 'winner_figs' and 'winner_coeffs'
		
		##########____Colorspaces_____######
		for iter in np.arange(3,25):
			pic_use=0 #used to input whether or not the picture was used into the coeff array
			############________COLORSPACES_______#############
			if iter==3:
				i_ext=img
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				name='Gray'
			elif iter==4:
				i_ext=img
				i_ext=cv2.transform(i_ext,m_1)
				name='BGR Blue'
			elif iter==5:
				i_ext=img
				i_ext=cv2.transform(i_ext,m_2)
				name='BGR Green'
			elif iter==6:
				i_ext=img
				i_ext=cv2.transform(i_ext,m_3)
				name='BGR Red'
			elif iter==7:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
				i_ext=cv2.transform(i_ext,m_1)
				name='CIE X'
			elif iter==8:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
				i_ext=cv2.transform(i_ext,m_2)
				name='CIE Y'
			elif iter==9:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
				i_ext=cv2.transform(i_ext,m_3)
				name='CIE Z'
			elif iter==10:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
				i_ext=cv2.transform(i_ext,m_1)
				name='YCrCb Y'
			elif iter==11:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
				i_ext=cv2.transform(i_ext,m_2)
				name='YCrCb Cr'
			elif iter==12:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
				i_ext=cv2.transform(i_ext,m_3)
				name='YCrCb Cb'
				#cv2.imwrite('YCrCb Cb test.jpg',i_ext)
			elif iter==13:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
				i_ext=cv2.transform(i_ext,m_1)
				name='HSV H'
			elif iter==14:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
				i_ext=cv2.transform(i_ext,m_2)
				name='HSV S'
			elif iter==15:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
				i_ext=cv2.transform(i_ext,m_3)
				name='HSV V'
			elif iter==16:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
				i_ext=cv2.transform(i_ext,m_1)
				name='HLS H'
			elif iter==17:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
				i_ext=cv2.transform(i_ext,m_2)
				name='HLS L'
			elif iter==18:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
				i_ext=cv2.transform(i_ext,m_3)
				name='HLS S'
			elif iter==19:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
				i_ext=cv2.transform(i_ext,m_1)		
				name='CLa L'
			elif iter==20:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
				i_ext=cv2.transform(i_ext,m_2)
				name='CLa a'
			elif iter==21:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
				i_ext=cv2.transform(i_ext,m_3)		
				name='CLa b'
			elif iter==22:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
				i_ext=cv2.transform(i_ext,m_1)
				name='CLu L'
			elif iter==23:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
				i_ext=cv2.transform(i_ext,m_2)	
				name='CLu u'
			elif iter==24:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
				i_ext=cv2.transform(i_ext,m_3)		
				name='CLu v'
			i_ext_t=cv2.adaptiveThreshold(i_ext,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,box,0)	
			i_ext_m=cv2.medianBlur(i_ext_t,3)
			
			var1=i_ext_m-i_ext
			var3=round(np.sum(np.square(var1),dtype=np.float64)*100/float(10000))
			var_agv=np.round(np.average(i_ext))
			#corr=np.corrcoef(np.array((var1)))[0,1]

			# pltfig.Figure(figsize=[12,10], dpi=200, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
			plt.subplot(4,6,iter)
			if thresh=='y':
				loss,i_ext_t=cv2.threshold(i_ext,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				plt.imshow(i_ext_t,cmap='gray')
				thresh_name='_t'
			else:
				# i_ext=cv2.equalizeHist(i_ext)
				plt.imshow(i_ext,cmap='gray')
				thresh_name=''
			plt.axis('off')
			plt.title(name+'\n'+str(var3)+" "+str(var_agv),fontsize=8)
		if pic_show=='y':
			plt.show()
		else:
			# plt.savefig('CS2_'+thresh_name+filename, bbox_inches='tight')
			plt.savefig('CS2_'+thresh_name+filename[:-4]+'.jpg', format='jpg', dpi=400)
		counter1=counter1+1
		print counter1/float(numfil)
