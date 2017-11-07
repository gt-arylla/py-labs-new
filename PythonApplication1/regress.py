
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
print 'running...'
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

maxAccuracy = 0.0
cb_best=0

test_number=20011

cb_list=[]
cct_list=[]
if test_number==0:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==2:
    cb_list=range(0,54)
    cct_list=range(0,54)
    cct_list=[39]
elif test_number==200:
    cb_list=range(0,54)
    cct_list=[7,8,18,26,33,39]
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-Org-PhotoSet'
elif test_number==2001:
    cb_list=range(0,54)
    cct_list=[39]
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-Org-PhotoSet'
elif test_number==20011:
    cb_list=range(0,54)
    cct_list=[39]
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-Org-PhotoSet'
elif test_number==222:
    cb_list=range(0,54)
    cct_list=[39]
    val_keep_list_list=[[0],[50],[100],[30],[70]]
    filter_col='0-Org-Yellow'
elif test_number==223:
    cb_list=range(0,54)
    cct_list=[39]
    val_keep_list_list=[[100],[200],[300],[325],[400],[450],[500],[600],[700],[900],[1300],[1400],[1500],[1600],[1700],[2500],[3000],[3500],[1000]]
    filter_col='0-Org-Lux'
elif test_number==21:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==211:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==212:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==213:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==22: #toggle turning on and off thickness and white count, use best CCT
    cb_list=range(0,54)
    cb=0
    cct_list=[[0,277+cb, 333, 334, 343+39],[1,277+cb, 333, 334, 343+39],[277+cb, 333, 334, 343+39]]
elif test_number==3:
    cb_list=range(0,54)
    cct_list=[0]


#Loop thru all cb bin vals
for cb in cb_list:
    for cct in cct_list:
        for val_keep_list_index in range(0,len(val_keep_list_list)):

            #read csv
            data = pd.read_csv('cbrgb_3.csv', header = 0)

            # data = data.dropna()
            #print(data.shape)
            # print(list(data.columns))

            ###remove rows as needed - based on column values
            #row_drop_list=[]
            #col_of_interest=332 #PhotoSet
            #for row in range(0,data.shape[0]-1):
            #    keep_row=False
            #    for val_keep in val_keep_list:
            #        if data.iloc[row,col_of_interest]==val_keep:
            #            keep_row=True
            #    if not keep_row:
            #        row_drop_list.append(row)
            #        print row_drop_list
            data=data.loc[data[filter_col].isin(val_keep_list_list[val_keep_list_index])]
            #data.drop(row_drop_list)
            #print(data.shape) 
        
            drop = range(data.shape[1])
            #Only keep Cb, LUX, Mark and Light Warmth columns
            mark_loc=0
            if test_number==0:
                drop = np.delete(drop, [770+cb, 830, 831, 834])
                mark_loc=2
            elif test_number==3: #drop white count
                drop = np.delete(drop, [277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=2
                #temperature starts at 343
            elif test_number==2:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 343+cct])
                mark_loc=4
                #temperature starts at 343
            elif test_number==200:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 343+cct])
                mark_loc=4
                #temperature starts at 343
            elif test_number==2001:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 337])
                mark_loc=4
                #temperature starts at 343
            elif test_number==20011:
                drop = np.delete(drop, [0,1,66,121,277+cb, 333, 334, 337])
                mark_loc=6
                #temperature starts at 343
            elif test_number==222:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 343+cct])
                mark_loc=4
                #temperature starts at 343
            elif test_number==223:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 343+cct])
                mark_loc=4
                #temperature starts at 343
            elif test_number==201: #drop thickness
                drop = np.delete(drop, [1,277+cb, 333, 334, 343+cct])
                mark_loc=3
                #temperature starts at 343
            elif test_number==21: #use yellow value instead of cct
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=4
                #temperature starts at 343
            elif test_number==211: #drop thickness
                drop = np.delete(drop, [1,277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=3
                #temperature starts at 343
            elif test_number==212: #drop white count
                drop = np.delete(drop, [0,277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=3
                #temperature starts at 343
            elif test_number==213: #drop white count
                drop = np.delete(drop, [277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=2
                #temperature starts at 343
            elif test_number==22: #use yellow value instead of cct
                drop = np.delete(drop, cct) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=4
                #temperature starts at 343
            data.drop(data.columns[drop], axis=1, inplace=True)
            #print(data.head(100))

            # sns.heatmap(data.corr())
            # plt.show()

            #Y data is the 4th column - mark
    

            y = data.iloc[:,mark_loc]
            #Drop the 4th col for X data
            data.drop(data.columns[[mark_loc]], axis=1, inplace=True)
            X = data.iloc[:,:]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=0)
            X_train.shape

            classifier = LogisticRegression(solver='newton-cg', random_state = 0)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_train)
            from sklearn.metrics import confusion_matrix
            confusion_matrix = confusion_matrix(y_train, y_pred)
           # print(confusion_matrix)

            print('Index CB CCT:,'+str(val_keep_list_index)+ ','+str(cb)+ ','+str(cct)+','+str(data.shape[0])+', {:.5f}'.format(classifier.score(X_train, y_train)))
            if (classifier.score(X_train, y_train) > maxAccuracy):
                #print maxAccuracy
                maxAccuracy = classifier.score(X_train, y_train)
                #print maxAccuracy
                #print cb_best
                cb_best=cb
                cct_best=cct
                val_keep_list_index_best=val_keep_list_index
                #print cb_best
                #print '*******************************************NEW cb BEST******************'
                #from sklearn.metrics import classification_report
                #print(classification_report(y_train, y_pred))

print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(maxAccuracy))
print 'best cb is' +str(cb_best)
print 'best cct is' +str(cct_best)
print 'best index is' +str(val_keep_list_index_best)
#print 'best cct is' +str(cb_best)