##Code and examples of useful Python and OpenCV processes

Morphological Transforms
    kernel = np.ones((1,1),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    dilation = cv2.dilate(img,kernel,iterations = 1)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

Resize
	img=cv2.resize(img,None,fx=resize_frac,fy=resize_frac,interpolation = cv2.INTER_CUBIC)
	img=cv2.resize(img,(width,height))

	Random
	float=np.random.random()

Export Image
	cv2.imwrite('test2.jpg',i_ext)


Thresholds
	#Adaptive Threshold
	img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,box,0)
	
	#Binary Threshold
	img=cv2.threshold(img,200,255,cv2.THRESH_BINARY)[1]
	
	#Otsu Threshold
	img=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

	
Clay Protocol
	clahe=cv2.createCLAHE(tileGridSize=(4,4))
	i_ext=clahe.apply(i_ext)
	
Draw Contours
	cv2.drawContours(img,contour,-1,(255,255,255),5)
	
	for cnt in np.arange(len(contour)):
		red=50+int(155*np.random.random())
		blue=50+int(155*np.random.random())
		green=50+int(155*np.random.random())
		i_shower=copy.copy(i)
		cv2.drawContours(i_shower, [contour[cnt]], -1, (red,green,blue), thickness=-1)
		
	cv2.circle(i,tuple(D_center),1,(0,255,0),100)
	
	cv2.rectangle(i_crop_rot,(x,y),(x+w,y+h),(0,255,0),2)

	
	blur = cv2.GaussianBlur(img,(5,5),0)

Saving
	cv2.imwrite('name,result2)
Colorspaces
	m_1=np.array([1,0,0]).reshape((1,3))
	m_2=np.array([0,1,0]).reshape((1,3))
	m_3=np.array([0,0,1]).reshape((1,3))
	
	i_ext=img
	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	name='Gray'

	i_ext=img
	i_ext=cv2.transform(i_ext,m_1)
	name='BGR Blue'

	i_ext=img
	i_ext=cv2.transform(i_ext,m_2)
	name='BGR Green'

	i_ext=img
	i_ext=cv2.transform(i_ext,m_3)
	name='BGR Red'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
	i_ext=cv2.transform(i_ext,m_1)
	name='CIE X'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
	i_ext=cv2.transform(i_ext,m_2)
	name='CIE Y'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
	i_ext=cv2.transform(i_ext,m_3)
	name='CIE Z'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	i_ext=cv2.transform(i_ext,m_1)
	name='YCrCb Y'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	i_ext=cv2.transform(i_ext,m_2)
	name='YCrCb Cr'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	i_ext=cv2.transform(i_ext,m_3)
	name='YCrCb Cb'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	i_ext=cv2.transform(i_ext,m_1)
	name='HSV H'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	i_ext=cv2.transform(i_ext,m_2)
	name='HSV S'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	i_ext=cv2.transform(i_ext,m_3)
	name='HSV V'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
	i_ext=cv2.transform(i_ext,m_1)
	name='HLS H'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
	i_ext=cv2.transform(i_ext,m_2)
	name='HLS L'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
	i_ext=cv2.transform(i_ext,m_3)
	name='HLS S'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
	i_ext=cv2.transform(i_ext,m_1)		
	name='CLa L'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
	i_ext=cv2.transform(i_ext,m_2)
	name='CLa a'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
	i_ext=cv2.transform(i_ext,m_3)		
	name='CLa b'


	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
	i_ext=cv2.transform(i_ext,m_1)
	name='CLu L'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
	i_ext=cv2.transform(i_ext,m_2)	
	name='CLu u'

	i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
	i_ext=cv2.transform(i_ext,m_3)		
	name='CLu v'
	
GitHub
	#Add
	#Adds everything
	git add .
	#Add specific file
	git add script.py
	
	#Commit
	git commit -m "Comments go here"
	
	#Push
	git push
	
	#Status
	#See what has been added, changed, etc
	git status
	
Rotate without crop
	theta=43
	diagonal=int(np.ceil(np.sqrt(img.shape[0]**2+img.shape[1]**2)))
	background=np.zeros([diagonal,diagonal,3],np.uint8)
	insert_point=[(diagonal-i.shape[0])/2,(diagonal-i.shape[1])/2]
	background[insert_point[0]:insert_point[0]+i.shape[0],insert_point[1]:insert_point[1]+i.shape[1]]=img
	rotation_kernel=cv2.getRotationMatrix2D((origin[0],origin[1]),theta,1)
	background=cv2.warpAffine(background,rotation_kernel,(background.shape[0],background.shape[1]))

Rotate
	rotation_kernel=cv2.getRotationMatrix2D((origin[0],origin[1]),theta,1)
	img=cv2.warpAffine(img,rotation_kernel,(img.shape[0],img.shape[1]))	

