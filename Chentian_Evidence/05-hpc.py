import pandas as pd
import pickle
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import losses
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

import matplotlib.pyplot as plt

test = pd.read_csv("../data/processed/test3.csv").drop(columns = ['Unnamed: 0'])
train = pd.read_csv("../data/processed/train3.csv").drop(columns = ['Unnamed: 0'])

X_train = train.drop(['dos','u2r','r2l','probe','normal'],axis=1)
X_test = test.drop(['dos','u2r','r2l','probe','normal'],axis=1)

y_train = train[['dos','u2r','r2l','probe','normal']]
y_test = test[['dos','u2r','r2l','probe','normal']]


def adversarial_attack(model, X, y, epsilon):
    
    logits_model = tf.keras.Model(model.input, model.layers[-1].output)

    adv_fgsm_x = fast_gradient_method(logits_model,
                                      X,
                                      epsilon,
                                      np.inf,
                                      targeted=False)

    adv_x = adv_fgsm_x
    result = model.evaluate(adv_x,y)
    print(result)
    metrics = dict(zip(model.metrics_names, result))
    
    return  metrics['loss'], metrics['accuracy'], metrics['auc'], adv_x
    

def create_model(X,y,layers,activation):
    model = Sequential()
    model.add(Dense(118, input_dim=118, activation=activation))
    for i in range(layers):
        model.add(Dense(118, activation=activation))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','AUC'])
    model.fit(X,y,epochs=5,batch_size=128)
    return model

def adversarial_training(X, y, adv, layers, activation):
    adv_x = tf.concat([tf.constant(X.astype('float32')),adv],0)
    adv_y = pd.concat([y,y])
    adv_model = create_model(adv_x, adv_y, layers, activation)
    return adv_model

epsilons = [0.01,0.1,1,10]
layers_num = [1,3,5,10]
activations = ['tanh', 'relu', 'swish']

def comparision(X_train, y_train, X_test, y_test):

    loss_org_train_list = []
    accuracy_org_train_list = []
    auc_org_train_list = []
    loss_org_test_list = []
    accuracy_org_test_list = []
    auc_org_test_list = []

    loss_attack1_train_list = []
    accuracy_attack1_train_list = []
    auc_attack1_train_list = []
    loss_attack1_test_list = []
    accuracy_attack1_test_list = []
    auc_attack1_test_list = []

    loss_new_train_list = []
    accuracy_new_train_list = []
    auc_new_train_list = []
    loss_new_test_list = []
    accuracy_new_test_list = []
    auc_new_test_list = []

    num=0
    for epsilon in epsilons:
        print('epsilon is '+str(epsilon))
        for layers in layers_num:
            for activation in activations:

                print(num)
                print('Create model:')
                print('Orginal model with '+str(layers)+' layers.')
                # model, loss_org_train, accuracy_org_train, auc_org_train = create_model(X_train, y_train,X_test, y_test, layers, activation)
                model = create_model(X_train, y_train, layers, activation)
                result = model.evaluate(X_train,y_train)
                print(result)
                metrics = dict(zip(model.metrics_names, result))
                loss_org_train = metrics['loss']
                accuracy_org_train = metrics['accuracy']
                auc_org_train = metrics['auc']

                result = model.evaluate(X_test,y_test)
                print(result)
                metrics = dict(zip(model.metrics_names, result))
                loss_org_test = metrics['loss']
                accuracy_org_test = metrics['accuracy']
                auc_org_test = metrics['auc']

                print('Adversarial attack on orginal model: ')
                print('activation function: '+activation)
                print('epsilon is '+str(epsilon))
                print('Adversarial attack on train set:')
                loss_attack1_train, accuracy_attack1_train,auc_attack1_train, adv_x = adversarial_attack(model, X_train, y_train,epsilon)
                print('Adversarial attack on test set:')
                loss_attack1_test, accuracy_attack1_test, auc_attack1_test,_ = adversarial_attack(model, X_test, y_test,epsilon)
                print('Adversarial training:')
                adv_model = adversarial_training(X_train, y_train, adv_x, layers, activation)
                print('Performance on trian set after adversarial training:')
                loss_new_train, accuracy_new_train, auc_new_train,_ = adversarial_attack(adv_model, X_train, y_train,epsilon)
                print('Performance on trian set after adversarial training:')
                loss_new_test, accuracy_new_test, auc_new_test,_ = adversarial_attack(adv_model, X_test, y_test,epsilon)

                loss_org_train_list.append(loss_org_train)
                accuracy_org_train_list.append(accuracy_org_train)
                auc_org_train_list.append(auc_org_train)
                loss_org_test_list.append(loss_org_test)
                accuracy_org_test_list.append(accuracy_org_test)
                auc_org_test_list.append(auc_org_test)

                loss_attack1_train_list.append(loss_attack1_train)
                accuracy_attack1_train_list.append(accuracy_attack1_train)
                auc_attack1_train_list.append(auc_attack1_train)        
                loss_attack1_test_list.append(loss_attack1_test)
                accuracy_attack1_test_list.append(accuracy_attack1_test)
                auc_attack1_test_list.append(auc_attack1_test)

                loss_new_train_list.append(loss_new_train)
                accuracy_new_train_list.append(accuracy_new_train)
                auc_new_train_list.append(auc_new_train)
                loss_new_test_list.append(loss_new_test)
                accuracy_new_test_list.append(accuracy_new_test)
                auc_new_test_list.append(auc_new_test)

                num+=1

    return loss_org_train_list, accuracy_org_train_list, auc_org_train_list, \
            loss_org_test_list, accuracy_org_test_list, auc_org_test_list, \
            loss_attack1_train_list, accuracy_attack1_train_list,auc_attack1_train_list, \
            loss_attack1_test_list, accuracy_attack1_test_list, auc_attack1_test_list, \
            loss_new_train_list, accuracy_new_train_list, auc_new_train_list, \
            loss_new_test_list, accuracy_new_test_list,auc_new_test_list 


loss_org_train_list, accuracy_org_train_list, auc_org_train_list, \
loss_org_test_list, accuracy_org_test_list, auc_org_test_list, \
loss_attack1_train_list, accuracy_attack1_train_list,auc_attack1_train_list, \
loss_attack1_test_list, accuracy_attack1_test_list, auc_attack1_test_list, \
loss_new_train_list, accuracy_new_train_list, auc_new_train_list, \
loss_new_test_list, accuracy_new_test_list,auc_new_test_list  = comparision(X_train, y_train, X_test, y_test)

metrics=[loss_org_train_list, accuracy_org_train_list, auc_org_train_list, \
            loss_org_test_list, accuracy_org_test_list, auc_org_test_list, \
            loss_attack1_train_list, accuracy_attack1_train_list,auc_attack1_train_list, \
            loss_attack1_test_list, accuracy_attack1_test_list, auc_attack1_test_list, \
            loss_new_train_list, accuracy_new_train_list, auc_new_train_list, \
            loss_new_test_list, accuracy_new_test_list,auc_new_test_list]

pickle.dump(metrics, open("../data/processed/metrics/metrics.p", "wb" ))
