from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import time
import statistics
import matplotlib.pyplot as plt
import pickle


def create_model(features,layers,activation):
    model = Sequential()
    model.add(Dense(features, input_dim=features, activation=activation))
    for i in range(layers):
        model.add(Dense(features, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','AUC'])
    return model
    
def train_model(train,test,model):
    train_X = train.drop('normal',axis=1)
    test_X = train.drop('normal',axis=1)
    train_Y = train['normal']
    test_Y = train['normal']
    old_score = 0
    test_score_acc=[]
    train_score_acc=[]
    test_score_auc=[]
    train_score_auc=[]
    model_time = []
    epoch = 0
    while (epoch < 2) or (test_score_auc[epoch-1] > test_score_auc[epoch-2]):
        model_time1 = time.perf_counter()
        model.fit(train_X,train_Y,epochs=1,batch_size=128)
        current_train_score = model.evaluate(train_X,train_Y)
        model_time2 = time.perf_counter()
        model_time.append(model_time2 - model_time1)
        train_score_acc.append(current_train_score[1])
        train_score_auc.append(current_train_score[2])
        current_test_score = model.evaluate(test_X,test_Y)
        test_score_acc.append(current_test_score[1])
        test_score_auc.append(current_test_score[2])
        epoch = epoch + 1
    avg_time = statistics.mean(model_time)
    return (test_score_acc[:-1], train_score_acc[:-1], test_score_auc[:-1], train_score_auc[:-1], avg_time)
    




test = pd.read_csv("../data/processed/test.csv").drop(columns = ['Unnamed: 0'])
train = pd.read_csv("../data/processed/train.csv").drop(columns = ['Unnamed: 0'])
features = train.shape[1] -1

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



score = train_model(train,test,model1_relu)

with open(../scratch/scores) as f:
	pickle.dump(score, f)




