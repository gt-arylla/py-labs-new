
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
print 'running...'
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

maxAccuracy = 0.0
cb_best = 0

test_number = 3

cb_list = []
cct_list = []
if test_number == 0:
    cb_list = range(0,54)
    cct_list = [0]
elif test_number == 2:
    cb_list = range(0,54)
    cct_list = range(0,54)
    cct_list = [39]
elif test_number == 200:
    cb_list = range(0,54)
    cct_list = [7,8,18,26,33,39]
    val_keep_list_list = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col = '0-Org-PhotoSet'
elif test_number == 2001:
    cb_list = range(0,54)
    cct_list = [39]
    val_keep_list_list = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col = '0-Org-PhotoSet'
elif test_number == 20011:
    cb_list = range(0,54)
    cct_list = [39]
    val_keep_list_list = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col = '0-Org-PhotoSet'
elif test_number == 222:
    cb_list = range(0,54)
    cct_list = [39]
    val_keep_list_list = [[0],[50],[100],[30],[70]]
    filter_col = '0-Org-Yellow'
elif test_number == 223:
    cb_list = range(0,54)
    cct_list = [39]
    val_keep_list_list = [[100],[200],[300],[325],[400],[450],[500],[600],[700],[900],[1300],[1400],[1500],[1600],[1700],[2500],[3000],[3500],[1000]]
    filter_col = '0-Org-Lux'
elif test_number == 21:
    cb_list = range(0,54)
    cct_list = [0]
elif test_number == 211:
    cb_list = range(0,54)
    cct_list = [0]
elif test_number == 212:
    cb_list = range(0,54)
    cct_list = [0]
elif test_number == 213:
    cb_list = range(0,54)
    cct_list = [0]
elif test_number == 22: #toggle turning on and off thickness and white count, use best CCT
    cb_list = range(0,54)
    cb = 0
    cct_list = [[0,277 + cb, 333, 334, 343 + 39],[1,277 + cb, 333, 334, 343 + 39],[277 + cb, 333, 334, 343 + 39]]
elif test_number == 3:
    cb_list = range(0,54)
    cb_list=[0]
    cct_list = [0]


#Loop thru all cb bin vals
for hue in range(0, 2):
    for sat in range(0, 2):
        for val in range(0, 1):
            for blue in range(0, 2):
                for green in range(0, 2):
                    for cb in range(0, 2):
                        for cct in range(0,5):
                            #if (p0 or pix or cb or p or r or lux or cct):
                            droppin=[]
                            droppin.append(0)#lux
                            #droppin.append(0) #pix
                            droppin.append(2)
                            
                            #droppin.append(1) #thick
                            #droppin.append(3)

                            droppin.append(4+cct)
                            droppin.append(18+cct)
                            droppin.append(32+cct)
                            droppin.append(46+cct)
                            
                            #if hue:
                            #    droppin.append(58+6)
                            #if sat:
                            #    droppin.append(59+6)
                            #if val:
                            #    droppin.append(60+6)
                            #if blue:
                            #    droppin.append(60+6)
                            #if green:
                            #    droppin.append(61+6)
                            #if cb:
                            #    droppin.append(62+6)
                            #if cct:
                            #    droppin.append(63+6)
                            #droppin.append(332+cb) #cb
                                
                            ##if p0:
                            #droppin.append(387)
                            ##if p10:
                            #droppin.append(388)
                            ##if p25:
                            ##    droppin.append(389)
                            ##if p50:
                            #droppin.append(390)
                            ##if p75:
                            ##    droppin.append(391)
                            ##if p90:
                            #droppin.append(392)
                            ##if p100:
                            ##    droppin.append(393)

                            ##if r0:
                            #droppin.append(394)
                            ##if r10:
                            #droppin.append(395)
                            ##if r25:
                            ##    droppin.append(396)
                                
                            #droppin.append(434) #cct
                                
                            #droppin.append(453) #lux

                            markNum=len(droppin)
                            droppin.append(70)
                            #print markNum
                                
       
                                           
                                            
                            #read csv
                            data = pd.read_csv('LuxSweep.csv', header = 0)

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
                            #data = data.loc[data[filter_col].isin(val_keep_list_list[val_keep_list_index])]
                            #data.drop(row_drop_list)
                            #print(data.shape)
        
                            drop = range(data.shape[1])
                            #Only keep Cb, LUX, Mark and Light Warmth columns
                            mark_loc = 0
                            if test_number == 0:
                                drop = np.delete(drop, [770 + cb, 830, 831, 834])
                                mark_loc = 2
                            elif test_number == 3: #drop white count
                                drop = np.delete(drop, droppin) #thick;white count;Cb;Lux;Mark;Yellow
                                mark_loc = markNum
                                #temperature starts at 343
                            elif test_number == 2:
                                drop = np.delete(drop, [0,1,277 + cb, 333, 334, 343 + cct])
                                mark_loc = 4
                                #temperature starts at 343
           
                            data.drop(data.columns[drop], axis=1, inplace=True)
                            #print(data.head(10))

                            # sns.heatmap(data.corr())
                            # plt.show()

                            #Y data is the 4th column - mark
    

                            y = data.iloc[:,mark_loc]
                            #Drop the 4th col for X data
                            data.drop(data.columns[[mark_loc]], axis=1, inplace=True)
                            X = data.iloc[:,:]

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.000, random_state=0)
                            X_train.shape

                            classifier = LogisticRegressionCV(solver='newton-cg', random_state = 0)
                            classifier.fit(X_train, y_train)

                            y_pred = classifier.predict(X_train)
                            from sklearn.metrics import confusion_matrix
                            confusion_matrix = confusion_matrix(y_train, y_pred)
                            
                            tp=confusion_matrix[0,0]
                            fn=confusion_matrix[0,1]
                            tn=confusion_matrix[1,1]
                            fp=confusion_matrix[1,0]
                            tpr=tp/float(tp+fn)
                            tnr=tn/float(tn+fp)
                            fpr=fp/float(fp+tn)
                            #truth=tpr-(1-tnr)
                            truth=tpr-fpr
                            # print(confusion_matrix)

                            #print 'Truth: ' + str(truth)
                            #print 'TPP: '+str(tpr)
                            #print 'fpr: '+str(fpr)


                            #print classifier.sparsify()
                            #print(confusion_matrix)
                            #print y_train.mean()
                            #print pd.DataFrame(zip(X.columns, np.transpose(classifier.coef_)))
                            #print classifier.coef_
                            #print classifier.intercept_

                            print 'Index CB CCT pix thick cb p r cct lux:,' +str(data.shape[0]) + ','+str(hue) + ','+ str(sat) + ','+ str(val) + ','+ str(blue) + ','+ str(green) + ','+ str(cb) + ','+ str(cct) +  ', {:.5f}'.format(classifier.score(X_train, y_train))+","+str(truth)+ "," + str(tpr) + "," + str(fpr) + "," + str(tp) + "," + str(fn) + "," + str(tn) + "," + str(fp)
                            #if (classifier.score(X_train, y_train) > maxAccuracy):
                            #    #print maxAccuracy
                            #    maxAccuracy = classifier.score(X_train, y_train)
                            #    #print maxAccuracy
                            #    #print cb_best
                            #    cb_best = cb
                            #    cct_best = cct
                            #    val_keep_list_index_best = val_keep_list_index
                            #    #print cb_best
                            #    #print '*******************************************NEW cb BEST******************'
                            #    #from sklearn.metrics import classification_report
                            #    #print(classification_report(y_train, y_pred))

print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(maxAccuracy))
print 'best cb is' + str(cb_best)
print 'best cct is' + str(cct_best)
print 'best index is' + str(val_keep_list_index_best)
#print 'best cct is' +str(cb_best)