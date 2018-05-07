#functions
import numpy as np
import cv2


def make_colorspace_single(img,colorspace_index):
    m_1=np.array([1,0,0]).reshape((1,3))
    m_2=np.array([0,1,0]).reshape((1,3))
    m_3=np.array([0,0,1]).reshape((1,3))
    name=''
    i_ext=img
    if colorspace_index==0:
        i_ext=img
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        name='Gray'
    elif colorspace_index==1:
        i_ext=img
        i_ext=cv2.transform(i_ext,m_1)
        name='BGR Blue'
    elif colorspace_index==2:
        i_ext=img
        i_ext=cv2.transform(i_ext,m_2)
        name='BGR Green'
    elif colorspace_index==3:
        i_ext=img
        i_ext=cv2.transform(i_ext,m_3)
        name='BGR Red'
    elif colorspace_index==4:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
        i_ext=cv2.transform(i_ext,m_1)
        name='CIE X'
    elif colorspace_index==5:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
        i_ext=cv2.transform(i_ext,m_2)
        name='CIE Y'
    elif colorspace_index==6:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
        i_ext=cv2.transform(i_ext,m_3)
        name='CIE Z'
    elif colorspace_index==7:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        i_ext=cv2.transform(i_ext,m_1)
        name='YCrCb Y'
    elif colorspace_index==8:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        i_ext=cv2.transform(i_ext,m_2)
        name='YCrCb Cr'
    elif colorspace_index==9:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        i_ext=cv2.transform(i_ext,m_3)
        name='YCrCb Cb'
    #cv2.imwrite('YCrCb Cb test.jpg',i_ext)
    elif colorspace_index==10:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        i_ext=cv2.transform(i_ext,m_1)
        name='HSV H'
    elif colorspace_index==11:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        i_ext=cv2.transform(i_ext,m_2)
        name='HSV S'
    elif colorspace_index==12:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        i_ext=cv2.transform(i_ext,m_3)
        name='HSV V'
    elif colorspace_index==13:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        i_ext=cv2.transform(i_ext,m_1)
        name='HLS H'
    elif colorspace_index==14:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        i_ext=cv2.transform(i_ext,m_2)
        name='HLS L'
    elif colorspace_index==15:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        i_ext=cv2.transform(i_ext,m_3)
        name='HLS S'
    elif colorspace_index==16:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        i_ext=cv2.transform(i_ext,m_1)		
        name='CLa L'
    elif colorspace_index==17:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        i_ext=cv2.transform(i_ext,m_2)
        name='CLa a'
    elif colorspace_index==18:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        i_ext=cv2.transform(i_ext,m_3)		
        name='CLa b'
    elif colorspace_index==19:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
        i_ext=cv2.transform(i_ext,m_1)
        name='CLu L'
    elif colorspace_index==20:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
        i_ext=cv2.transform(i_ext,m_2)	
        name='CLu u'
    elif colorspace_index==21:
        i_ext=cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
        i_ext=cv2.transform(i_ext,m_3)		
        name='CLu v'
    return i_ext,name
