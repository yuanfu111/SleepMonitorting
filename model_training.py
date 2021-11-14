import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pickle
import os
def get_training_data(path):
    # try one participant
    X_path = path + "/physiological"
    X_dirs = os.listdir(X_path)
    Y_path = path + "/processed_annotation"

    data_raw_df = pd.read_csv(X_path + "/" + X_dirs[0])
    anno_raw_df = pd.read_csv(Y_path + "/" + X_dirs[0][:-4] + "_processed.csv")
    data_df = pd.DataFrame(data_raw_df, columns=["gsr"])
    data_extracted_df = np.array(feature_extraction(data_df, 50))
    # anno_df = pd.DataFrame(anno_raw_df, columns=["arousal","activeness"])
    anno_df = pd.DataFrame(anno_raw_df, columns=["activeness"])

    for i in range(1,len(X_dirs)):
        print(X_path+ "/"+X_dirs[i])
        print(Y_path + "/"+ X_dirs[i][:-4] +"_processed.csv")
        data_raw_df_new  = pd.read_csv(X_path+ "/"+X_dirs[1])
        anno_raw_df_new  = pd.read_csv(Y_path + "/"+ X_dirs[1][:-4] +"_processed.csv")
        data_df_new = pd.DataFrame(data_raw_df_new, columns=["gsr"])
        # anno_df = pd.DataFrame(anno_raw_df_new, columns=["arousal","activeness"])
        anno_df_new = pd.DataFrame(anno_raw_df_new, columns=["activeness"])
        data_extracted_df_new = np.array(feature_extraction(data_df_new, 50))
        data_extracted_df = np.vstack((data_extracted_df,data_extracted_df_new))
        anno_df = anno_df.append(anno_df_new)

    Y = np.array(anno_df)

    return data_extracted_df,Y

def feature_extraction(raw_data,window_size):
    # only mean
    raw_data_grouped = raw_data.groupby(lambda x: math.floor(x / window_size))
    return raw_data_grouped.mean()
# def model(x,w,b):
#     return tf.multiply(x,w) + b

# def linear_regression_training(X,y):
#     model = Model()
#
#     EPOCHS = 20
#     LEARNING_RATE = 0.1
#
#     for epoch in range(EPOCHS):  # 迭代次数
#         with tf.GradientTape() as tape:  # 追踪梯度
#             loss = loss_fn(model, X, y)  # 计算损失
#         dW, db = tape.gradient(loss, [model.W, model.b])  # 计算梯度
#         model.W.assign_sub(LEARNING_RATE * dW)  # 更新梯度
#         model.b.assign_sub(LEARNING_RATE * db)
#         # 输出计算过程
#         print('Epoch [{}/{}], loss [{:.3f}], W/b [{:.3f}/{:.3f}]'.format(epoch, EPOCHS, loss,
#                                                                          float(model.W.numpy()),
#                                                                          float(model.b.numpy())))
#     plt.scatter(X, y)
#     plt.plot(X, model(X), c='r')
#     plt.show()
# def loss_fn(model, x, y):
#     y_ = model(x)
#     return tf.reduce_mean(tf.square(y_ - y))
# class Model(object):
#     def __init__(self):
#         self.W = tf.Variable(tf.random.uniform([1]))  # 随机初始化参数
#         self.b = tf.Variable(tf.random.uniform([1]))
#
#     def __call__(self, x):
#         return self.W * x + self.b  # w*x + b
def knn(X_train, Y_train,X_test, Y_test):

    #classification
    k = 15
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    # test
    assess_model_socre = knn.score(X_test, Y_test)
    print('Test set score for KNN n = {}:{:2f}'.format(k, assess_model_socre))
    joblib.dump(knn, 'knn.pkl')
def decision_tree(X_train, Y_train,X_test, Y_test):
    tree_clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random", max_depth=15
                                           , min_samples_leaf=10
                                           , min_samples_split=10)
    tree_clf = tree_clf.fit(X_train, Y_train)
    result = tree_clf.score(X_test, Y_test)
    print("Decision tree:", result)
    joblib.dump(tree_clf, 'decition_tree.pkl')

if __name__ == "__main__":
    training_data_path = "./training_data"
    X,Y = get_training_data(training_data_path)
    X = X.reshape(-1,1)
    print(X.shape,Y.shape)
    # Normalization
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    print(X_scaled)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, shuffle=True)
    print("X train size: {}, Y train size: {}".format(len(X_train), len(Y_train)))
    print("X test size: {}, Y test size: {}".format(len(X_test), len(Y_test)))
    decision_tree(X_train, Y_train,X_test, Y_test)