from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import time
import statistics
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def create_model(features,layers,activation):
    model = Sequential()
    model.add(Dense(features, input_dim=features, activation=activation))
    for i in range(layers):
        model.add(Dense(features, activation=activation))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model
    
def train_model(train,test,model,num):
    train_X = train.drop(['normal','dos','u2r','r2l','probe'],axis=1)
    test_X = train.drop(['normal','dos','u2r','r2l','probe'],axis=1)
    train_Y = train.loc[:,['normal','dos','u2r','r2l','probe']]
    test_Y = train.loc[:,['normal','dos','u2r','r2l','probe']]
    old_score = 0
    test_score_acc=[]
    train_score_acc=[]
    test_score_auc=[]
    train_score_auc=[]
    model_time = []
    test_norm_mat=[]
    test_dos_mat=[]
    test_u2r_mat=[]
    test_r2l_mat=[]
    test_probe_mat=[]
    epoch = 0
    while (epoch < 2) or (test_score_auc[epoch-1] > test_score_auc[epoch-2]):
        model_time1 = time.perf_counter()
        model.fit(train_X,train_Y,epochs=1,batch_size=128)
        model_time2 = time.perf_counter()
        model_time.append(model_time2 - model_time1)
        yhat = model.predict(train_X)
        yhat = yhat.round()
        train_score_acc.append(accuracy_score(train_Y,yhat))
        train_score_auc.append(roc_auc_score(train_Y,yhat))
        yhat = model.predict(test_X)
        yhat = yhat.round()
        yhat1=[[] for _ in range(5)]
        for i in range(len(yhat)):
            for j in range(5):
                yhat1[j].append(yhat[i][j])
        test_score_acc.append(accuracy_score(test_Y,yhat))
        test_score_auc.append(roc_auc_score(test_Y,yhat))
        test_norm_mat=confusion_matrix(test_Y['normal'],yhat1[0])
        test_dos_mat=confusion_matrix(test_Y['dos'],yhat1[1])
        test_u2r_mat=confusion_matrix(test_Y['u2r'],yhat1[2])
        test_r2l_mat=confusion_matrix(test_Y['r2l'],yhat1[3])
        test_probe_mat=confusion_matrix(test_Y['probe'],yhat1[4])
        epoch = epoch + 1
    avg_time = statistics.mean(model_time[:-1])
    total_time=sum(model_time[:-1])
    model.save('model'+str(num))
    return (test_score_acc[:-1], train_score_acc[:-1], test_score_auc[:-1], train_score_auc[:-1], avg_time, total_time, test_norm_mat[:-1], test_dos_mat[:-1], test_u2r_mat[:-1], test_r2l_mat[:-1], test_probe_mat[:-1])
    




test = pd.read_csv("/user/work/zg18997/test1.csv").drop(columns = ['Unnamed: 0'])
train = pd.read_csv("/user/work/zg18997/train1.csv").drop(columns = ['Unnamed: 0'])
features = train.shape[1] -5

model1_tanh = create_model(features,1,'tanh')
model1_relu = create_model(features,1,'relu')
model1_swish = create_model(features,1,'swish')
model3_tanh = create_model(features,3,'tanh')
model3_relu = create_model(features,3,'relu')
model3_swish = create_model(features,3,'swish')
model5_tanh = create_model(features,5,'tanh')
model5_relu = create_model(features,5,'relu')
model5_swish = create_model(features,5,'swish')
model10_tanh = create_model(features,10,'tanh')
model10_relu = create_model(features,10,'relu')
model10_swish = create_model(features,10,'swish')


score=[]
score.append(train_model(train,test,model1_tanh,1))
score.append(train_model(train,test,model1_relu,2))
score.append(train_model(train,test,model1_swish,3))
score.append(train_model(train,test,model3_tanh,4))
score.append(train_model(train,test,model3_relu,5))
score.append(train_model(train,test,model3_swish,6))
score.append(train_model(train,test,model5_tanh,7))
score.append(train_model(train,test,model5_relu,8))
score.append(train_model(train,test,model5_swish,9))
score.append(train_model(train,test,model10_tanh,10))
score.append(train_model(train,test,model10_relu,11))
score.append(train_model(train,test,model10_swish,12))
with open('/user/work/zg18997/scores2','wb') as f:
	pickle.dump(score, f)




