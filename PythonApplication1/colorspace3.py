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
resize=1
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
	directory='G:\Developer\py-labs\FigureImport'
#directory="G://Google Drive//Original Images//Embroidery//171130 TriPhotoTest MaskingSubset//"

print directory

#default values are yes, no.  If user gives a third console input, use default values
if len(sys.argv)>2:
	thresh='n'
	eq_hist='n'
	clay='n'
	s3='n'
	pic_show='y'
else:
	thresh=raw_input('Apply thresholds? y/n ')
	if thresh=='n':
		eq_hist=raw_input('Equalize Histogram? y/n ')
		if eq_hist=='n':
			clay=raw_input('Apply Clay Protocol? y/n ')
			if clay=='y':
				s3=raw_input('With Clay post-process? y/n ')
				r_check='n'
	# r_check=raw_input('Resize Image? y/n ')
	pic_show=raw_input('Show Pic? y/n ')
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
			
		pltfig.Figure(figsize=[12,10], dpi=200, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
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
		# print box
		############___________PREP EXTRACTIONS_______############
		m_1=np.array([1,0,0]).reshape((1,3))
		m_2=np.array([0,1,0]).reshape((1,3))
		m_3=np.array([0,0,1]).reshape((1,3))
		
		tight_range=[130,150]
		counter0=0 #use for the winner counter, to input images into 'winner_figs' and 'winner_coeffs'
		
		##########____Colorspaces_____######
		for iter in np.arange(8):
			pic_use=0 #used to input whether or not the picture was used into the coeff array
			############________COLORSPACES_______#############
			if iter==0:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
				i_ext=cv2.transform(i_ext,m_3)
				i_ext=np.bitwise_not(i_ext)
				name='YCrCb Cb'
				i_ext_temp=cv2.equalizeHist(i_ext)
				cv2.imwrite("export.jpg",i_ext_temp)
				cv2.imwrite("basis.jpg",img);
				plt.figure(5)
				plt.imshow(i_ext,clim=(tight_range[0],tight_range[1]),cmap="jet")
				plt.savefig("plt_export.jpg",dpi=500)
				
				plt.figure(6)
				plt.imshow(i_ext,clim=(tight_range[0],tight_range[1]),cmap="jet")
				plt.show()
				
				# plt.subplot(121)
				# plt.imshow(i_ext)
				# plt.subplot(122)
				# plt.hist(i_ext.ravel(),256,[0,256])
				# plt.show()
			elif iter==1:
				i_ext=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
				i_ext=cv2.transform(i_ext,m_1)
				name='HSV H'
			elif iter==2:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
				i_ext=cv2.transform(i_ext,m_2)
				name='HSV S'
				# plt.show()
				# plt.imshow(i_ext)
				# plt.show()
			elif iter==3:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				i_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				# i_ext=cv2.transform(i_ext,m_1)
				name='Original'
			elif iter==4:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
				i_ext=cv2.transform(i_ext,m_2)
				name='CLa a'
			elif iter==5:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
				i_ext=cv2.transform(i_ext,m_3)		
				name='CLa b'
			elif iter==6:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
				i_ext=cv2.transform(i_ext,m_2)	
				name='CLu u'
			elif iter==7:
				i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
				i_ext=cv2.transform(i_ext,m_3)		
				name='CLu v'
			if iter!=3:
				if eq_hist=='y':
					i_ext=cv2.equalizeHist(i_ext)
				if clay=='y':
					clahe=cv2.createCLAHE(tileGridSize=(4,4))
					i_ext=clahe.apply(i_ext)
				if s3=='y':
					i_ext=cv2.medianBlur(i_ext,43)
					i_ext=cv2.adaptiveThreshold(i_ext,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,box,0)
					
					
				i_ext_b=cv2.GaussianBlur(i_ext,(11,11),0)
				i_ext_t=cv2.adaptiveThreshold(i_ext_b,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,box,0)	
				kernel = np.ones((4,4),np.uint8)
				i_ext_t = cv2.erode(i_ext_t,kernel,iterations = 1)
				
				i_ext_m=cv2.medianBlur(i_ext,3)
				i_ext_b=cv2.GaussianBlur(i_ext,(11,11),0)
				
				var1=i_ext_b-i_ext
				var3=round(np.sum(np.square(var1),dtype=np.float64)*100/float(10000))
				var_agv=np.round(np.average(i_ext))
				#corr=np.corrcoef(np.array((var1)))[0,1]

				pltfig.Figure(figsize=(12,10), dpi=200, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
				plt.subplot(2,4,iter+1)
				
				if thresh=='y':
					plt.imshow(i_ext_t)
					thresh_name='_t'
				else:
					plt.imshow(i_ext, clim=(tight_range[0],tight_range[1]),cmap="jet")
					thresh_name=''
				plt.axis('off')
				plt.title(name+'\n'+str(var3)+" "+str(var_agv),fontsize=8)
			else:
				pltfig.Figure(figsize=(12,10), dpi=200, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
				plt.subplot(2,4,iter+1)
				plt.imshow(i_ext)
				plt.axis('off')
				avg=int(np.average(i_grey))
				print filname+","+str(avg)
				plt.title(avg,fontsize=8)
		
		if pic_show=='y':
			plt.show()
		else:
			# dummy=1
			plt.savefig('CS3pure_'+thresh_name+filename, format='jpg', dpi=200)
		counter1=counter1+1
		# print counter1/float(numfil)
