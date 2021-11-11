#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.signal import filtfilt
from scipy import signal

import numpy as np


# In[2]:


data_raw_df = pd.read_csv("./dataset_sub1/physiologica_lsub_2.csv")
anno_raw_df = pd.read_csv("./dataset_sub1/processed_/sub_2_processed.csv")
data_df = pd.DataFrame(data_raw_df,columns=["gsr","emg_zygo","emg_coru","emg_trap","bvp"])
anno_df = pd.DataFrame(anno_raw_df,columns=["valence","arousal","video","emo"])


# In[3]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns.values.tolist():
        mean_value = df[feature_name].mean()
        std_value = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / (std_value)
    return result


# In[4]:


plt.plot(data_df["gsr"])
plt.xlabel("gsr")


# In[5]:


plt.plot(data_df["emg_coru"])
plt.xlabel("emg_coru")


# In[6]:


# filtering
# first-order butterworth
fs = 1000
fc = 50
Wn = fc/(fs/2)
order = 1
b, a = signal.butter(order, Wn, 'low')
data_filtered = pd.DataFrame(index=data_df.index,columns=data_df.columns)
for i in range(data_df.shape[1]):
    data_filtered.iloc[:,i] = filtfilt(b, a, data_df.iloc[:,i])

plt.plot(data_filtered["gsr"])
plt.xlabel("gsr_filtered")


# In[8]:


normalized_anno_df = normalize(anno_df)

# valence: degree of being pleasant arousal:degree of excitedness

plt.scatter(normalized_anno_df["arousal"],normalized_anno_df["valence"],label='valence',c = anno_df["video"])
plt.xlabel("arousal")
plt.ylabel("valence")


# In[9]:


# define the windows size 50ms in raw data, sampling frequency = 1000Hz
n =50 # ms
N = int(data_filtered.shape[0]/n) # number of data 


# In[10]:


# group the data
data_to_analysis = pd.DataFrame(data_filtered,columns=["gsr","emg_zygo","emg_coru","emg_trap","bvp"])# eliminate the time column
raw_data_grouped = data_to_analysis.groupby(lambda x: math.floor(x/n))
# group the annotation 
anno_to_analysis = pd.DataFrame(anno_raw_df,columns=["valence","arousal","emo"])
anno_grouped = anno_to_analysis.groupby(lambda x: math.floor(x/(n/50)))
anno_df = anno_grouped.mean()
#print(N,anno_df.shape[0])


# # Feature Extraction

# In[13]:


# Look at the mean
data_mean = raw_data_grouped.mean()

data_mean.plot()
plt.show()
plt.plot(data_mean['bvp'])
plt.show()
# linear correlation analysis
for phys_name in data_mean.columns.values.tolist():
    for anno_name in anno_df.columns.values.tolist():
        x = data_mean[phys_name]
        y = anno_df[anno_name]
        print(phys_name,anno_name)
        # pearson
        r,p = stats.pearsonr(x,y)  #
        print('r = %6.3f，p = %6.3f'%(r,p)) # if|r|<0.3 they are not linear correlated
#         # spearman
#         rho, pval = stats.spearmanr(x, y)
#         print('rho = %6.3f，pval = %6.3f'%(rho,pval))
#         # kendall 
#         tau, p_value = stats.kendalltau(x,y)
#         print('tau = %6.3f，p_value = %6.3f'%(tau,p_value))


# In[14]:


# the median
data_median = raw_data_grouped.median()
data_median.plot()
for phys_name in data_median.columns.values.tolist():
    for anno_name in anno_df.columns.values.tolist():
        x = data_median[phys_name]
        y = anno_df[anno_name]
        print(phys_name,anno_name)
        # pearson
        r,p = stats.pearsonr(x,y)  #
        print('r = %6.3f，p = %6.3f'%(r,p)) # if|r|<0.3 they are not linear correlated


# In[15]:


# the std
data_std = raw_data_grouped.std()
plt.plot(data_std["gsr"])
plt.xlabel("gsr std")
plt.show()
plt.plot(data_std["emg_zygo"])
plt.xlabel("emg_zygo std")
plt.show()

for phys_name in data_std.columns.values.tolist():
    for anno_name in anno_df.columns.values.tolist():
        x = data_std[phys_name]
        y = anno_df[anno_name]
        print(phys_name,anno_name)
        # pearson
        r,p = stats.pearsonr(x,y)  #
        print('r = %6.3f，p = %6.3f'%(r,p)) # if|r|<0.3 they are not linear correlated


# In[16]:


# the min
data_min = raw_data_grouped.min()
data_min.plot()
for phys_name in data_min.columns.values.tolist():
    for anno_name in anno_df.columns.values.tolist():
        x = data_min[phys_name]
        y = anno_df[anno_name]
        print(phys_name,anno_name)
        # pearson
        r,p = stats.pearsonr(x,y)  #
        print('r = %6.3f，p = %6.3f'%(r,p)) # if|r|<0.3 they are not linear correlated


# In[17]:


# the max
data_max = raw_data_grouped.max()
data_max.plot()
for phys_name in data_max.columns.values.tolist():
    for anno_name in anno_df.columns.values.tolist():
        x = data_max[phys_name]
        y = anno_df[anno_name]
        print(phys_name,anno_name)
        # pearson
        r,p = stats.pearsonr(x,y)  #
        print('r = %6.3f，p = %6.3f'%(r,p)) # if|r|<0.3 they are not linear correlated


# In[18]:


# the max-min
data_ex_dev = raw_data_grouped.max()-raw_data_grouped.min()
plt.plot(data_ex_dev["gsr"])
plt.xlabel("gsr ex_dev")
plt.show()
plt.plot(data_ex_dev["emg_zygo"])
plt.xlabel("emg_zygo ex_dev")
plt.show()

for phys_name in data_ex_dev.columns.values.tolist():
    for anno_name in anno_df.columns.values.tolist():
        x = data_ex_dev[phys_name]
        y = anno_df[anno_name]
        print(phys_name,anno_name)
        # pearson
        r,p = stats.pearsonr(x,y)  #
        print('r = %6.3f，p = %6.3f'%(r,p)) # if|r|<0.3 they are not linear correlated


# In[19]:


# the skew
data_skew = raw_data_grouped.skew()
plt.plot(data_skew["gsr"])
plt.xlabel("gsr skew")
plt.show()
plt.plot(data_skew["emg_zygo"])
plt.xlabel("emg_zygo skew")
plt.show()

for phys_name in data_skew.columns.values.tolist():
    for anno_name in anno_df.columns.values.tolist():
        x = data_skew[phys_name]
        y = anno_df[anno_name]
        print(phys_name,anno_name)
        # pearson
        r,p = stats.pearsonr(x,y)  #
        print('r = %6.3f，p = %6.3f'%(r,p)) # if|r|<0.3 they are not linear correlated


# In[20]:


# the kurtosis
data_kurtosis = raw_data_grouped.apply(pd.DataFrame.kurt)
plt.plot(data_kurtosis["gsr"])
plt.xlabel("gsr kurtosis")
plt.show()
plt.plot(data_kurtosis["emg_zygo"])
plt.xlabel("emg_zygo kurtosis")
plt.show()
for phys_name in data_kurtosis.columns.values.tolist():
    for anno_name in anno_df.columns.values.tolist():
        x = data_kurtosis[phys_name]
        y = anno_df[anno_name]
        print(phys_name,anno_name)
        # pearson
        r,p = stats.pearsonr(x,y)  #
        print('r = %6.3f，p = %6.3f'%(r,p)) # if|r|<0.3 they are not linear correlated


# In[44]:


# valence: degree of being pleasant
plt.plot(anno_df["valence"],label='valence')

# arousal:degree of excitedness
plt.plot(anno_df["arousal"],label="arousal")

plt.plot(anno_df["valence"]+anno_df["arousal"],label="v + a")
plt.legend()


# In[17]:


# valence: degree of being pleasant
plt.scatter(anno_df["valence"],anno_df["arousal"],c = anno_df["emo"])
plt.show()
# plt.legend()


# # Feature Selection

# In[21]:




concact_data = pd.concat([data_mean["gsr"],data_mean["emg_coru"],data_ex_dev["gsr"],data_ex_dev["emg_coru"],data_std["gsr"],data_std["emg_coru"],data_kurtosis['gsr'], data_skew['emg_coru'], data_skew['gsr'],data_kurtosis['emg_coru']],axis=1,keys=["gsr_mean","emg_coru_mean","gsr_ex_dev","emg_coru_ex_dev","gsr_std","emg_coru_std","gsr_kurt","emg_kurt","gsr_skew","emg_skew"])
X_raw = np.array(concact_data)
Y = np.array(anno_df["emo"])


# In[22]:


# Feature selection
from sklearn.feature_selection import VarianceThreshold
sel=VarianceThreshold(1)
sel.fit(X_raw)   #获得方差，不需要y
X = sel.transform(X_raw)
print('Variances is %s'%sel.variances_)
print('After transform is \n%s'%X)
print('The surport is %s'%sel.get_support(True))#如果为True那么返回的是被选中的特征的下标
print('The surport is %s'%sel.get_support(False))#如果为FALSE那么返回的是布尔类型的列表，反应是否选中这列特征


# from sklearn.feature_selection import SelectKBest,f_classif
# print('before transform:\n',X_raw)
# sel=SelectKBest(score_func=f_classif,k=4)
# sel.fit(X_raw,Y)  #计算统计指标，这里一定用到y
# X = sel.transform(X_raw)
# print('scores_:\n',sel.scores_)
# print('pvalues_:',sel.pvalues_)
# print('selected index:',sel.get_support(True))
# print('after transform:\n',X)


# In[23]:


plt.scatter(X[:,0],X[:,2],c=Y)
plt.show()


# # Classification

# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree   
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Normalization

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)


# In[25]:


# split train set and test set
X_train, X_test, Y_train, Y_test =train_test_split(X_scaled, Y, test_size=0.2, shuffle=True)
print("X train size: {}, Y train size: {}".format(len(X_train), len(Y_train)))
print("X test size: {}, Y test size: {}".format(len(X_test), len(Y_test)))


# In[26]:


# KNN

k = 15
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)
# test
assess_model_socre=knn.score(X_test,Y_test)
print('Test set score for KNN n = {}:{:2f}'.format(k,assess_model_socre))


# In[19]:


# parameter tuning-grid searcg

param_grid = [
    {
        "weights":["uniform"],
        "n_neighbors":[i for i in range(1,30)]
    },
    {
        "weights":["distance"],
        "n_neighbors":[i for i in range(1,30)],
        'p':[i for i in range(1,6)]
    }
     
]
knn_clf = KNeighborsClassifier()
knn_clf = GridSearchCV(knn_clf,param_grid=param_grid,cv=3)

knn_clf.fit(X_train,Y_train)

assess_model_socre=knn_clf.score(X_test,Y_test)
print('Test set score:{:2f}'.format(assess_model_socre))
knn_clf.best_estimator_ 
#p = 1 表示选择曼哈顿距离

#p = 2 表示选择欧拉距离（默认）

#p >=3 表示选择其他距离


# In[27]:


# Let's try decision tree


tree_clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter="random",max_depth=15
                                  ,min_samples_leaf=10
                                  ,min_samples_split=10)                  
tree_clf = tree_clf.fit(X_train,Y_train)                              
result = tree_clf.score(X_test,Y_test) 
print("Decision tree:",result)


# In[ ]:


# Try svc

svc_clf = SVC(kernel='poly') #rbf? poly?
svc_clf.fit(X_train, Y_train)
y_predict = svc_clf.predict(X_test)

# print (svc_clf.score(X_train, Y_train)) 
# print ('training set accuracy：', accuracy_score(Y_train, svc_clf.predict(X_train)))
# print (svc_clf.score(X_test, Y_test))
print ('test set accuracy：', accuracy_score(Y_test,y_predict)) # accuracy: linear - 0.50, rbf - 0.54 , poly - 0.52


# In[38]:


grid = GridSearchCV(SVC(), param_grid={'kernel':['linear','rbf','poly'],'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}, cv=4)
grid.fit(X_train, Y_train)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# In[567]:


# Let's try frequency analysis with FFT
from scipy.fftpack import fft,ifft
import numpy as np
from matplotlib.pylab import mpl
y = data_mean["gsr"][100:200]
N = len(y)
fs = 1000
f = [fs/(N-1)*n for n in range(0,N)]

Y = np.fft.fft(y)*2/N #*2/N 反映了FFT变换的结果与实际信号幅值之间的关系
absY = [np.abs(x) for x in Y] #求傅里叶变换结果的模

plt.plot(f[1:-1],absY[1:-1])

plt.xlabel('freq(Hz)')

plt.title("fft")
plt.show()


# In[590]:


import random
# for i,value in enumerate(data_mean["gsr"]):
#     if value == 0:
#         value = 0.0001
#     data_mean["gsr"][i] = 1/value
data_mean_normalized = normalize(data_mean)


# In[607]:


N = len(data_mean["gsr"])
sample_num = 1000
indeces = random.sample(range(0,N),sample_num)
indeces.sort()
data_sampled = data_mean_normalized.iloc[indeces]
anno_sampled = anno_df.iloc[indeces]

# normailization


# In[570]:


print(data_mean_normalized.iloc[indeces])


# In[ ]:





# In[608]:


plt.scatter(anno_sampled["arousal"],anno_sampled["valence"],c =anno_sampled["video"] )


# In[572]:


plt.scatter(anno_sampled["valence"],data_sampled["gsr"])


# In[573]:


plt.scatter(data_sampled["emg_zygo"],anno_sampled["arousal"])


# In[574]:


plt.scatter(data_sampled["emg_zygo"],anno_sampled["valence"])


# In[438]:


import matplotlib.pyplot as mp, seaborn
# std
# data_anno = pd.concat([data_mean[["gsr","emg_coru"]],anno_df],axis=1)
# df_corr = data_anno.corr()
# seaborn.heatmap(df_corr,center = 0, annot = True, cmap = "YlGnBu")
# mp.show()


# In[439]:



# # std
# data_anno = pd.concat([data_ex_dev[["gsr","emg_coru"]],anno_df],axis=1)
# df_corr = data_anno.corr()
# seaborn.heatmap(df_corr,center = 0, annot = True, cmap = "YlGnBu")
# mp.show()


# In[440]:



# # std
# data_anno = pd.concat([data_std[["gsr","emg_coru"]],anno_df],axis=1)
# df_corr = data_anno.corr()
# seaborn.heatmap(df_corr,center = 0, annot = True, cmap = "YlGnBu")
# mp.show()


# In[579]:


concact_data = pd.concat([data_mean["gsr"],data_mean["emg_coru"],data_ex_dev["gsr"],data_ex_dev["emg_coru"],data_std["gsr"],data_std["emg_coru"]],axis=1,keys=["gsr_mean","emg_coru_mean","gsr_ex_dev","emg_coru_ex_dev","gsr_std","emg_coru_std"])
data_anno = pd.concat([concact_data,anno_df],axis=1)
df_corr = data_anno.corr()
seaborn.heatmap(df_corr,center = 0, annot = True, cmap = "YlGnBu")
mp.show()


# In[580]:


# let's try regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
def stdError_func(y_test, y):
  return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
  return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
  y_mean = np.array(y)
  y_mean[:] = y.mean()
  return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)


# In[ ]:





# In[581]:


# multi-varialble linear fitting

x = np.array(concact_data.values)

y = np.array(anno_df["valence"].values)

cft = linear_model.LinearRegression()
print(x.shape)
cft.fit(x, y) #

print("model coefficients", cft.coef_)
print("model intercept", cft.intercept_)


predict_y =  cft.predict(x)
strError = stdError_func(predict_y, y)
R2_1 = R2_1_func(predict_y, y)
R2_2 = R2_2_func(predict_y, y)
score = cft.score(x, y) ##

print('strError={:.2f}, R2_1={:.2f},  R2_2={:.2f}, clf.score={:.2f}'.format(
    strError,R2_1,R2_2,score))


# In[444]:



poly_reg =PolynomialFeatures(degree=2) #三次多项式
X_ploy =poly_reg.fit_transform(x)
lin_reg_2=linear_model.LinearRegression()
lin_reg_2.fit(X_ploy,y)
predict_y =  lin_reg_2.predict(X_ploy)
strError = stdError_func(predict_y, y)
R2_1 = R2_1_func(predict_y, y)
R2_2 = R2_2_func(predict_y, y)
score = lin_reg_2.score(X_ploy, y) ##sklearn中自带的模型评估，与R2_1逻辑相同

print("coefficients", lin_reg_2.coef_)
print("intercept", lin_reg_2.intercept_)
print('degree={}: strError={:.2f}, R2_1={:.2f},  R2_2={:.2f}, clf.score={:.2f}'.format(
    3, strError,R2_1,R2_2,score))


# In[522]:


# feature selection
from sklearn import feature_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit
import sys
X = np.array(concact_data)
Y = np.array(anno_df["valence"])
names =concact_data.columns.values.tolist()

rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
print(sorted(scores, reverse=True))


# In[523]:


from sklearn.model_selection import train_test_split
# split train set and test set X Y
X_train, X_test, Y_train, Y_test =train_test_split(XNew, Y, test_size=0.2, shuffle=True)
print("X train size: {}, Y train size: {}".format(len(X_train), len(Y_train)))
print("X test size: {}, Y test size: {}".format(len(X_test), len(Y_test)))


# In[524]:


X_train.shape


# In[525]:


# try svm
from sklearn.svm import SVR
svr = SVR(kernel='linear')
svr.fit(X_train,Y_train)
svr_y_predict = svr.predict(X_test)


# In[526]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print("r2 score",r2_score(Y_test,svr_y_predict))


# In[527]:


r = len(X_test) + 1
plt.plot(np.arange(1,r), svr_y_predict, 'go-', label="predict")
plt.plot(np.arange(1,r), Y_test, 'co-', label="real")
plt.legend()
plt.show()


# In[ ]:




