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
    #separate the test and train datasets 
    train_X = train.drop(['normal','dos','u2r','r2l','probe'],axis=1)
    test_X = train.drop(['normal','dos','u2r','r2l','probe'],axis=1)
    train_Y = train.loc[:,['normal','dos','u2r','r2l','probe']]
    test_Y = train.loc[:,['normal','dos','u2r','r2l','probe']]
    old_score = 0 #used to track the AUC score of the last epoch
    #initialise the model metrics
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
    while (epoch < 2) or (test_score_auc[epoch-1] > test_score_auc[epoch-2]): #repeat until the test AUC score of the latest iteration is lower than the previous epoch
        model_time1 = time.perf_counter() #check the time
        model.fit(train_X,train_Y,epochs=1,batch_size=128)
        model_time2 = time.perf_counter() #check the time
        model_time.append(model_time2 - model_time1) #calculates the difference in the times to see how long the model took to train
        yhat = model.predict(train_X) #predicts the classification of the train dataset
        yhat = yhat.round() #round to classify the data
        train_score_acc.append(accuracy_score(train_Y,yhat)) #calculate accuracy of the prediction
        train_score_auc.append(roc_auc_score(train_Y,yhat)) #calculate AUC of the prediction
        yhat = model.predict(test_X) #predicts the classification of the test dataset
        yhat = yhat.round() #round to classify the data
        yhat1=[[] for _ in range(5)] #create a matrix reformat the prediction
        #reformat the model prediction so we can separate the attack types
        for i in range(len(yhat)): 
            for j in range(5):
                yhat1[j].append(yhat[i][j])
        test_score_acc.append(accuracy_score(test_Y,yhat)) #calculate accuracy of the prediction
        test_score_auc.append(roc_auc_score(test_Y,yhat)) #calculate AUC of the prediction
        #calculate the confusion matrix for each of the classifications
        test_norm_mat=confusion_matrix(test_Y['normal'],yhat1[0])
        test_dos_mat=confusion_matrix(test_Y['dos'],yhat1[1])
        test_u2r_mat=confusion_matrix(test_Y['u2r'],yhat1[2])
        test_r2l_mat=confusion_matrix(test_Y['r2l'],yhat1[3])
        test_probe_mat=confusion_matrix(test_Y['probe'],yhat1[4])
        epoch = epoch + 1
    #manipulate the model times to useful metrics
    avg_time = statistics.mean(model_time[:-1])
    total_time=sum(model_time[:-1])
    model.save('model'+str(num)) #save models for the extension
    return (test_score_acc[:-1], train_score_acc[:-1], test_score_auc[:-1], train_score_auc[:-1], avg_time, total_time, test_norm_mat[:-1], test_dos_mat[:-1], test_u2r_mat[:-1], test_r2l_mat[:-1], test_probe_mat[:-1])
    




test = pd.read_csv("/user/work/zg18997/test2.csv").drop(columns = ['Unnamed: 0'])
train = pd.read_csv("/user/work/zg18997/train2.csv").drop(columns = ['Unnamed: 0'])
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




