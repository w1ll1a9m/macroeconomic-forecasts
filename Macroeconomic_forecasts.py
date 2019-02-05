#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 22:33:38 2019

@author: williamlopez
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from crowd_layers import CrowdsRegression, MaskedMultiMSE
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras import initializers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

np.random.seed(1)

def get_data(indicator):
    
    target_name1 = 't1_'+indicator
    target_name2 = 't2_'+indicator
    target_name3 = 't3_'+indicator
    target_name4 = 't4_'+indicator
    
    #experts data
    
 
    Experts_file = indicator+'_forecasts.xlsx'
    Data_exp = pd.read_excel(Experts_file);
    Data_exp = Data_exp[Data_exp['YQ'] != 196804];
    Ex_pred_1 = Data_exp[['YQ','ID', 'forecast1']]
    Ex_pred_1 =Ex_pred_1.pivot_table(index=['YQ'], columns=['ID'], values= 'forecast1')
    Ex_pred_2 = Data_exp[['YQ','ID', 'forecast2']]
    Ex_pred_2 =Ex_pred_2.pivot_table(index=['YQ'], columns=['ID'], values= 'forecast2')
    Ex_pred_3 = Data_exp[['YQ','ID', 'forecast3']]
    Ex_pred_3 =Ex_pred_3.pivot_table(index=['YQ'], columns=['ID'], values= 'forecast3')
    Ex_pred_4 = Data_exp[['YQ','ID', 'forecast4']]
    Ex_pred_4 =Ex_pred_4.pivot_table(index=['YQ'], columns=['ID'], values= 'forecast4')
    
    Ex_pred_1=Ex_pred_1.reset_index(drop=False)
    Ex_pred_2=Ex_pred_2.reset_index(drop=False)
    Ex_pred_3=Ex_pred_3.reset_index(drop=False)
    Ex_pred_4=Ex_pred_4.reset_index(drop=False)
    
    
    #All variables
    
    Data_x = pd.read_excel('X_ALL_PREVIOUS_Y.xlsx');
    Data_x1 = Data_x[['YQ','t1_RGDP','t1_UNEMP', 't1_CPROF','t1_HOUSING', 't1_INDPROD','t1_NGDP','t1_PGDP']]
    #Data_x1 = X.drop(target_name1, axis=1)
    #Data_x=Data_x.set_index('YQ')
    
    #targets for the time horizons
    
    Data_y = pd.read_excel('Y_ALL.xlsx');
    Data_y = Data_y[Data_y['YQ'] != 196804];
    Y1 = Data_y[['YQ',target_name1]]
    Y2 = Data_y[['YQ',target_name2]]
    Y3 = Data_y[['YQ',target_name3]]
    Y4 = Data_y[['YQ',target_name4]]
    
    # outlier detection
    
    Y1=iqr_check(Y1, target_name1)
    Y2=iqr_check(Y2, target_name2)
    Y3=iqr_check(Y3, target_name3)
    Y4=iqr_check(Y4, target_name4)
    
    # dropping  missing targets in the time horizons
    
    X1 = Data_x1[Data_x['YQ'].isin(Y1['YQ'])].reset_index(drop=True);
    X2 = Data_x1[Data_x['YQ'].isin(Y2['YQ'])].reset_index(drop=True);
    X3 = Data_x1[Data_x['YQ'].isin(Y3['YQ'])].reset_index(drop=True);
    X4 = Data_x1[Data_x['YQ'].isin(Y4['YQ'])].reset_index(drop=True);
    
    Ex1 = Ex_pred_1[Ex_pred_1['YQ'].isin(Y1['YQ'])].reset_index(drop=True);
    Ex2 = Ex_pred_2[Ex_pred_2['YQ'].isin(Y2['YQ'])].reset_index(drop=True);
    Ex3 = Ex_pred_3[Ex_pred_3['YQ'].isin(Y3['YQ'])].reset_index(drop=True);
    Ex4 = Ex_pred_4[Ex_pred_4['YQ'].isin(Y4['YQ'])].reset_index(drop=True);
       
    Y1=Y1.set_index('YQ')
    Y2=Y2.set_index('YQ')
    Y3=Y3.set_index('YQ')
    Y4=Y4.set_index('YQ')
    X1=X1.set_index('YQ')
    X2=X2.set_index('YQ')
    X3=X3.set_index('YQ')
    X4=X4.set_index('YQ')
    Ex1=Ex1.set_index('YQ')
    Ex2=Ex2.set_index('YQ')
    Ex3=Ex3.set_index('YQ')
    Ex4=Ex4.set_index('YQ')
    
    #mean prediction for each time horizon
   
    Mean_pred_1=Ex1.mean(axis=1)
    Mean_pred_1=Mean_pred_1.reset_index(drop=True)
    Mean_pred_2=Ex2.mean(axis=1)
    Mean_pred_2=Mean_pred_2.reset_index(drop=True)
    Mean_pred_3=Ex3.mean(axis=1)
    Mean_pred_3=Mean_pred_3.reset_index(drop=True)
    Mean_pred_4=Ex4.mean(axis=1)
    Mean_pred_4=Mean_pred_4.reset_index(drop=True)
    
    #replacing missing values on the predictions for the crowd layer
    Ex1[np.isnan(Ex1)]=999999999
    Ex2[np.isnan(Ex2)]=999999999
    Ex3[np.isnan(Ex3)]=999999999
    Ex4[np.isnan(Ex4)]=999999999
    
    #getting the variables
    
    N_exp_1=Ex1.shape[1]
    N_exp_2=Ex2.shape[1]
    N_exp_3=Ex3.shape[1]
    N_exp_4=Ex4.shape[1]
    N_points_1=Y1.shape[0]
    N_points_2=Y2.shape[0]
    N_points_3=Y3.shape[0]
    N_points_4=Y4.shape[0]
    
    #from data frames to values
    
    Ex_pred_1=Ex_pred_1.values[:,0:N_exp_1]
    Ex_pred_2=Ex_pred_2.values[:,0:N_exp_2]
    Ex_pred_3=Ex_pred_3.values[:,0:N_exp_3]
    Ex_pred_4=Ex_pred_4.values[:,0:N_exp_4]
    X1=X1.values[:,0:350]
    X2=X2.values[:,0:350]
    X3=X3.values[:,0:350]
    X4=X4.values[:,0:350]
    Y1=Y1.values
    Y2=Y2.values
    Y3=Y3.values
    Y4=Y4.values
    Mean_pred_1=Mean_pred_1.values
    Mean_pred_2=Mean_pred_2.values
    Mean_pred_3=Mean_pred_3.values
    Mean_pred_4=Mean_pred_4.values
    
    return X1, X2, X3, X4, Ex1, Ex2, Ex3, Ex4, Y1, Y2, Y3, Y4, Mean_pred_1, Mean_pred_2, Mean_pred_3, Mean_pred_4, N_exp_1,  N_exp_2,  N_exp_3,  N_exp_4, N_points_1, N_points_2, N_points_3, N_points_4 



def iqr_check(Y, target1):
    
    q75, q25 = np.percentile(Y[target1], [75, 25]);
    iqr = q75 - q25;
    w1 = q25 - 5*iqr; #changed this as 5
    w2 = q75 + 5*iqr; #changed this as 5
    
    #I dont want to eliminate high shifts, I want to eliminate adjustment changes 
    
    Y[target1] = Y.apply(lambda x: x[target1] if x[target1] >= w1 and x[target1] <= w2 else 999999, axis=1);
    
    #MANUAL CHECK!
    Y[target1] = Y.apply(lambda x: x[target1] if abs(x[target1]) <= 90 else 999999, axis=1);
    Y = Y[Y[target1] != 999999].reset_index(drop=True);
    
    return Y;



def build_base_model():
    
    inputs = Input(shape=(7,))

    # Dense Layers and output layer
    x = Dense(3, activation='relu',kernel_initializer=initializers.he_normal(seed=1))(inputs)
    x = Dense(7, activation='relu',kernel_initializer=initializers.he_normal(seed=1))(x)
    #x = Dense(7, activation='relu',kernel_initializer=initializers.he_normal(seed=1))(x)
    x = Dense(3, activation='relu',kernel_initializer=initializers.he_normal(seed=1))(x)
    preds = Dense(1, activation='linear')(x)
    
    #creating the model and compiling it
    model = Model(inputs=inputs, outputs=preds)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'] )
    
    
    return model, inputs, preds;


def Base_nn(x_train, y_train, x_test, N_exp, mode):
    
    Weights_crowdlayer_RGDP=0
    
    # creating the NN
    np.random.seed(1)
    RGDP_nn, RGDP_in, RGDP_preds = build_base_model()
    
    if mode == 'crowd':
        
        # add crowds layer on top of the base model
        ma_preds = CrowdsRegression(N_exp, conn_type="B")(RGDP_preds)
        
        # instantiate specialized masked loss to handle missing answers
        loss = MaskedMultiMSE().loss
        
        # compile model with masked loss and train
        RGDP_nn = Model(RGDP_in, ma_preds)
        RGDP_nn.compile(optimizer='adam', loss=loss, metrics=['mae', 'mse'])
    
    # training the models
    
    #model.summary()
    
    #model fit
    
    history_RGDP = RGDP_nn.fit(x_train, y_train, epochs=600, batch_size=10,  verbose=2, validation_split=0.02)
    RGDP_nn.save_weights("RGDP_1.h5")
    
    print(history_RGDP.history.keys())
    # "Loss
#    plt.plot(history_RGDP.history['loss'])
#    plt.plot(history_RGDP.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'validation'], loc='upper left')
#    plt.show()
    
    
    # remove crowd layer before making predictions
    
    # make predictions
    
    # save weights from crowds layer for later
    if mode == 'crowd':
        Weights_crowdlayer_RGDP = RGDP_nn.layers[4].get_weights()
    
    # skip CrowdsLayer for predictions
    RGDP_nn = Model(RGDP_in, RGDP_preds) 
    RGDP_nn.compile(loss="mse", optimizer='adam')
    
    
    
    RGDP_nn_pred = RGDP_nn.predict(x_test)
    
    
    
    
    return RGDP_nn_pred, Weights_crowdlayer_RGDP

    
    
def model_evaluation(y_pred, mean_pred, y_test):
    
    mse_model = mean_squared_error(y_test, y_pred)
    mae_model = mean_absolute_error(y_test, y_pred)
    mse_mean_experts = mean_squared_error(y_test, mean_pred)
    mae_mean_experts = mean_absolute_error(y_test, mean_pred)
    
    return mse_model, mae_model, mse_mean_experts, mae_mean_experts
    
def plot_preds(y_test, mean_pred, y_pred_cw, y_pred_nn):
    
    plt.plot(y_test, color='black')
    plt.plot(mean_pred, color='green')
    plt.plot(y_pred_cw, color='red')
    plt.plot(y_pred_nn, color='blue')
    plt.show()
    
def print_error(mse_nn_cl, mae_nn_cl, mse_ex, mae_ex, indicator, mse_nn, mae_nn):
    
    print('Indicator: ' , indicator)
    print('MSE NN: ' , mse_nn)
    print('MAE NN: ' , mae_nn)
    print('MSE NN+CL: ' , mse_nn_cl)
    print('MAE NN+CL: ' , mae_nn_cl)
    print('MSE Experts: ' , mse_ex)
    print('MAE Experts: ' , mae_ex)
    

    
  #%%

#Getting the data

RGDP_x1, RGDP_x2, RGDP_x3, RGDP_x4, RGDP_Ex_pred_1, RGDP_Ex_pred_2, RGDP_Ex_pred_3, RGDP_Ex_pred_4, RGDP_y1, RGDP_y2, RGDP_y3, RGDP_y4, RGDP_Mean_pred_1, RGDP_Mean_pred_2, RGDP_Mean_pred_3, RGDP_Mean_pred_4, RGDP_N_exp_1, RGDP_N_exp_2, RGDP_N_exp_3, RGDP_N_exp_4, RGDP_N_points_1, RGDP_N_points_2, RGDP_N_points_3, RGDP_N_points_4 = get_data('RGDP')
UNEMP_x1, UNEMP_x2, UNEMP_x3, UNEMP_x4, UNEMP_Ex_pred_1, UNEMP_Ex_pred_2, UNEMP_Ex_pred_3, UNEMP_Ex_pred_4, UNEMP_y1, UNEMP_y2, UNEMP_y3, UNEMP_y4, UNEMP_Mean_pred_1, UNEMP_Mean_pred_2, UNEMP_Mean_pred_3, UNEMP_Mean_pred_4, UNEMP_N_exp_1, UNEMP_N_exp_2, UNEMP_N_exp_3, UNEMP_N_exp_4, UNEMP_N_points_1, UNEMP_N_points_2, UNEMP_N_points_3, UNEMP_N_points_4 = get_data('UNEMP')
CPROF_x1, CPROF_x2, CPROF_x3, CPROF_x4, CPROF_Ex_pred_1, CPROF_Ex_pred_2, CPROF_Ex_pred_3, CPROF_Ex_pred_4, CPROF_y1, CPROF_y2, CPROF_y3, CPROF_y4, CPROF_Mean_pred_1, CPROF_Mean_pred_2, CPROF_Mean_pred_3, CPROF_Mean_pred_4, CPROF_N_exp_1, CPROF_N_exp_2, CPROF_N_exp_3, CPROF_N_exp_4, CPROF_N_points_1, CPROF_N_points_2, CPROF_N_points_3, CPROF_N_points_4 = get_data('CPROF')
HOUSING_x1, HOUSING_x2, HOUSING_x3, HOUSING_x4, HOUSING_Ex_pred_1, HOUSING_Ex_pred_2, HOUSING_Ex_pred_3, HOUSING_Ex_pred_4, HOUSING_y1, HOUSING_y2, HOUSING_y3, HOUSING_y4, HOUSING_Mean_pred_1, HOUSING_Mean_pred_2, HOUSING_Mean_pred_3, HOUSING_Mean_pred_4, HOUSING_N_exp_1, HOUSING_N_exp_2, HOUSING_N_exp_3, HOUSING_N_exp_4, HOUSING_N_points_1, HOUSING_N_points_2, HOUSING_N_points_3, HOUSING_N_points_4 = get_data('HOUSING')
INDPROD_x1, INDPROD_x2, INDPROD_x3, INDPROD_x4, INDPROD_Ex_pred_1, INDPROD_Ex_pred_2, INDPROD_Ex_pred_3, INDPROD_Ex_pred_4, INDPROD_y1, INDPROD_y2, INDPROD_y3, INDPROD_y4, INDPROD_Mean_pred_1, INDPROD_Mean_pred_2, INDPROD_Mean_pred_3, INDPROD_Mean_pred_4, INDPROD_N_exp_1, INDPROD_N_exp_2, INDPROD_N_exp_3, INDPROD_N_exp_4, INDPROD_N_points_1, INDPROD_N_points_2, INDPROD_N_points_3, INDPROD_N_points_4 = get_data('INDPROD')
NGDP_x1, NGDP_x2, NGDP_x3, NGDP_x4, NGDP_Ex_pred_1, NGDP_Ex_pred_2, NGDP_Ex_pred_3, NGDP_Ex_pred_4, NGDP_y1, NGDP_y2, NGDP_y3, NGDP_y4, NGDP_Mean_pred_1, NGDP_Mean_pred_2, NGDP_Mean_pred_3, NGDP_Mean_pred_4, NGDP_N_exp_1, NGDP_N_exp_2, NGDP_N_exp_3, NGDP_N_exp_4, NGDP_N_points_1, NGDP_N_points_2, NGDP_N_points_3, NGDP_N_points_4 = get_data('NGDP')
PGDP_x1, PGDP_x2, PGDP_x3, PGDP_x4, PGDP_Ex_pred_1, PGDP_Ex_pred_2, PGDP_Ex_pred_3, PGDP_Ex_pred_4, PGDP_y1, PGDP_y2, PGDP_y3, PGDP_y4, PGDP_Mean_pred_1, PGDP_Mean_pred_2, PGDP_Mean_pred_3, PGDP_Mean_pred_4, PGDP_N_exp_1, PGDP_N_exp_2, PGDP_N_exp_3, PGDP_N_exp_4, PGDP_N_points_1, PGDP_N_points_2, PGDP_N_points_3, PGDP_N_points_4 = get_data('PGDP')

#train test split

RGDP_x1_tr, RGDP_x1_tst, RGDP_y1_tr, RGDP_y1_tst = train_test_split(RGDP_x1, RGDP_y1, test_size=0.1, shuffle=False)
RGDP_x2_tr, RGDP_x2_tst, RGDP_y2_tr, RGDP_y2_tst = train_test_split(RGDP_x2, RGDP_y2, test_size=0.1, shuffle=False)
RGDP_x3_tr, RGDP_x3_tst, RGDP_y3_tr, RGDP_y3_tst = train_test_split(RGDP_x3, RGDP_y3, test_size=0.1, shuffle=False)
RGDP_x4_tr, RGDP_x4_tst, RGDP_y4_tr, RGDP_y4_tst = train_test_split(RGDP_x4, RGDP_y4, test_size=0.1, shuffle=False)

UNEMP_x1_tr, UNEMP_x1_tst, UNEMP_y1_tr, UNEMP_y1_tst = train_test_split(UNEMP_x1, UNEMP_y1, test_size=0.1, shuffle=False)
UNEMP_x2_tr, UNEMP_x2_tst, UNEMP_y2_tr, UNEMP_y2_tst = train_test_split(UNEMP_x2, UNEMP_y2, test_size=0.1, shuffle=False)
UNEMP_x3_tr, UNEMP_x3_tst, UNEMP_y3_tr, UNEMP_y3_tst = train_test_split(UNEMP_x3, UNEMP_y3, test_size=0.1, shuffle=False)
UNEMP_x4_tr, UNEMP_x4_tst, UNEMP_y4_tr, UNEMP_y4_tst = train_test_split(UNEMP_x4, UNEMP_y4, test_size=0.1, shuffle=False)

CPROF_x1_tr, CPROF_x1_tst, CPROF_y1_tr, CPROF_y1_tst = train_test_split(CPROF_x1, CPROF_y1, test_size=0.1, shuffle=False)
CPROF_x2_tr, CPROF_x2_tst, CPROF_y2_tr, CPROF_y2_tst = train_test_split(CPROF_x2, CPROF_y2, test_size=0.1, shuffle=False)
CPROF_x3_tr, CPROF_x3_tst, CPROF_y3_tr, CPROF_y3_tst = train_test_split(CPROF_x3, CPROF_y3, test_size=0.1, shuffle=False)
CPROF_x4_tr, CPROF_x4_tst, CPROF_y4_tr, CPROF_y4_tst = train_test_split(CPROF_x4, CPROF_y4, test_size=0.1, shuffle=False)

HOUSING_x1_tr, HOUSING_x1_tst, HOUSING_y1_tr, HOUSING_y1_tst = train_test_split(HOUSING_x1, HOUSING_y1, test_size=0.1, shuffle=False)
HOUSING_x2_tr, HOUSING_x2_tst, HOUSING_y2_tr, HOUSING_y2_tst = train_test_split(HOUSING_x2, HOUSING_y2, test_size=0.1, shuffle=False)
HOUSING_x3_tr, HOUSING_x3_tst, HOUSING_y3_tr, HOUSING_y3_tst = train_test_split(HOUSING_x3, HOUSING_y3, test_size=0.1, shuffle=False)
HOUSING_x4_tr, HOUSING_x4_tst, HOUSING_y4_tr, HOUSING_y4_tst = train_test_split(HOUSING_x4, HOUSING_y4, test_size=0.1, shuffle=False)

INDPROD_x1_tr, INDPROD_x1_tst, INDPROD_y1_tr, INDPROD_y1_tst = train_test_split(INDPROD_x1, INDPROD_y1, test_size=0.1, shuffle=False)
INDPROD_x2_tr, INDPROD_x2_tst, INDPROD_y2_tr, INDPROD_y2_tst = train_test_split(INDPROD_x2, INDPROD_y2, test_size=0.1, shuffle=False)
INDPROD_x3_tr, INDPROD_x3_tst, INDPROD_y3_tr, INDPROD_y3_tst = train_test_split(INDPROD_x3, INDPROD_y3, test_size=0.1, shuffle=False)
INDPROD_x4_tr, INDPROD_x4_tst, INDPROD_y4_tr, INDPROD_y4_tst = train_test_split(INDPROD_x4, INDPROD_y4, test_size=0.1, shuffle=False)

NGDP_x1_tr, NGDP_x1_tst, NGDP_y1_tr, NGDP_y1_tst = train_test_split(NGDP_x1, NGDP_y1, test_size=0.1, shuffle=False)
NGDP_x2_tr, NGDP_x2_tst, NGDP_y2_tr, NGDP_y2_tst = train_test_split(NGDP_x2, NGDP_y2, test_size=0.1, shuffle=False)
NGDP_x3_tr, NGDP_x3_tst, NGDP_y3_tr, NGDP_y3_tst = train_test_split(NGDP_x3, NGDP_y3, test_size=0.1, shuffle=False)
NGDP_x4_tr, NGDP_x4_tst, NGDP_y4_tr, NGDP_y4_tst = train_test_split(NGDP_x4, NGDP_y4, test_size=0.1, shuffle=False)

PGDP_x1_tr, PGDP_x1_tst, PGDP_y1_tr, PGDP_y1_tst = train_test_split(PGDP_x1, PGDP_y1, test_size=0.1, shuffle=False)
PGDP_x2_tr, PGDP_x2_tst, PGDP_y2_tr, PGDP_y2_tst = train_test_split(PGDP_x2, PGDP_y2, test_size=0.1, shuffle=False)
PGDP_x3_tr, PGDP_x3_tst, PGDP_y3_tr, PGDP_y3_tst = train_test_split(PGDP_x3, PGDP_y3, test_size=0.1, shuffle=False)
PGDP_x4_tr, PGDP_x4_tst, PGDP_y4_tr, PGDP_y4_tst = train_test_split(PGDP_x4, PGDP_y4, test_size=0.1, shuffle=False)

RGDP_Mean_pred_1_tr, RGDP_Mean_pred_1_tst = train_test_split(RGDP_Mean_pred_1, test_size=0.1, shuffle=False)
RGDP_Mean_pred_2_tr, RGDP_Mean_pred_2_tst = train_test_split(RGDP_Mean_pred_2, test_size=0.1, shuffle=False)
RGDP_Mean_pred_3_tr, RGDP_Mean_pred_3_tst = train_test_split(RGDP_Mean_pred_3, test_size=0.1, shuffle=False)
RGDP_Mean_pred_4_tr, RGDP_Mean_pred_4_tst = train_test_split(RGDP_Mean_pred_4, test_size=0.1, shuffle=False)

UNEMP_Mean_pred_1_tr, UNEMP_Mean_pred_1_tst = train_test_split(UNEMP_Mean_pred_1, test_size=0.1, shuffle=False)
UNEMP_Mean_pred_2_tr, UNEMP_Mean_pred_2_tst = train_test_split(UNEMP_Mean_pred_2, test_size=0.1, shuffle=False)
UNEMP_Mean_pred_3_tr, UNEMP_Mean_pred_3_tst = train_test_split(UNEMP_Mean_pred_3, test_size=0.1, shuffle=False)
UNEMP_Mean_pred_4_tr, UNEMP_Mean_pred_4_tst = train_test_split(UNEMP_Mean_pred_4, test_size=0.1, shuffle=False)

CPROF_Mean_pred_1_tr, CPROF_Mean_pred_1_tst = train_test_split(CPROF_Mean_pred_1, test_size=0.1, shuffle=False)
CPROF_Mean_pred_2_tr, CPROF_Mean_pred_2_tst = train_test_split(CPROF_Mean_pred_2, test_size=0.1, shuffle=False)
CPROF_Mean_pred_3_tr, CPROF_Mean_pred_3_tst = train_test_split(CPROF_Mean_pred_3, test_size=0.1, shuffle=False)
CPROF_Mean_pred_4_tr, CPROF_Mean_pred_4_tst = train_test_split(CPROF_Mean_pred_4, test_size=0.1, shuffle=False)

HOUSING_Mean_pred_1_tr, HOUSING_Mean_pred_1_tst = train_test_split(HOUSING_Mean_pred_1, test_size=0.1, shuffle=False)
HOUSING_Mean_pred_2_tr, HOUSING_Mean_pred_2_tst = train_test_split(HOUSING_Mean_pred_2, test_size=0.1, shuffle=False)
HOUSING_Mean_pred_3_tr, HOUSING_Mean_pred_3_tst = train_test_split(HOUSING_Mean_pred_3, test_size=0.1, shuffle=False)
HOUSING_Mean_pred_4_tr, HOUSING_Mean_pred_4_tst = train_test_split(HOUSING_Mean_pred_4, test_size=0.1, shuffle=False)

INDPROD_Mean_pred_1_tr, INDPROD_Mean_pred_1_tst = train_test_split(INDPROD_Mean_pred_1, test_size=0.1, shuffle=False)
INDPROD_Mean_pred_2_tr, INDPROD_Mean_pred_2_tst = train_test_split(INDPROD_Mean_pred_2, test_size=0.1, shuffle=False)
INDPROD_Mean_pred_3_tr, INDPROD_Mean_pred_3_tst = train_test_split(INDPROD_Mean_pred_3, test_size=0.1, shuffle=False)
INDPROD_Mean_pred_4_tr, INDPROD_Mean_pred_4_tst = train_test_split(INDPROD_Mean_pred_4, test_size=0.1, shuffle=False)

NGDP_Mean_pred_1_tr, NGDP_Mean_pred_1_tst = train_test_split(NGDP_Mean_pred_1, test_size=0.1, shuffle=False)
NGDP_Mean_pred_2_tr, NGDP_Mean_pred_2_tst = train_test_split(NGDP_Mean_pred_2, test_size=0.1, shuffle=False)
NGDP_Mean_pred_3_tr, NGDP_Mean_pred_3_tst = train_test_split(NGDP_Mean_pred_3, test_size=0.1, shuffle=False)
NGDP_Mean_pred_4_tr, NGDP_Mean_pred_4_tst = train_test_split(NGDP_Mean_pred_4, test_size=0.1, shuffle=False)

PGDP_Mean_pred_1_tr, PGDP_Mean_pred_1_tst = train_test_split(PGDP_Mean_pred_1, test_size=0.1, shuffle=False)
PGDP_Mean_pred_2_tr, PGDP_Mean_pred_2_tst = train_test_split(PGDP_Mean_pred_2, test_size=0.1, shuffle=False)
PGDP_Mean_pred_3_tr, PGDP_Mean_pred_3_tst = train_test_split(PGDP_Mean_pred_3, test_size=0.1, shuffle=False)
PGDP_Mean_pred_4_tr, PGDP_Mean_pred_4_tst = train_test_split(PGDP_Mean_pred_4, test_size=0.1, shuffle=False)

RGDP_Ex_pred_1_tr, RGDP_Ex_pred_1_tst = train_test_split(RGDP_Ex_pred_1, test_size=0.1, shuffle=False)
RGDP_Ex_pred_2_tr, RGDP_Ex_pred_2_tst = train_test_split(RGDP_Ex_pred_2, test_size=0.1, shuffle=False)
RGDP_Ex_pred_3_tr, RGDP_Ex_pred_3_tst = train_test_split(RGDP_Ex_pred_3, test_size=0.1, shuffle=False)
RGDP_Ex_pred_4_tr, RGDP_Ex_pred_4_tst = train_test_split(RGDP_Ex_pred_4, test_size=0.1, shuffle=False)

UNEMP_Ex_pred_1_tr, UNEMP_Ex_pred_1_tst = train_test_split(UNEMP_Ex_pred_1, test_size=0.1, shuffle=False)
UNEMP_Ex_pred_2_tr, UNEMP_Ex_pred_2_tst = train_test_split(UNEMP_Ex_pred_2, test_size=0.1, shuffle=False)
UNEMP_Ex_pred_3_tr, UNEMP_Ex_pred_3_tst = train_test_split(UNEMP_Ex_pred_3, test_size=0.1, shuffle=False)
UNEMP_Ex_pred_4_tr, UNEMP_Ex_pred_4_tst = train_test_split(UNEMP_Ex_pred_4, test_size=0.1, shuffle=False)

CPROF_Ex_pred_1_tr, CPROF_Ex_pred_1_tst = train_test_split(CPROF_Ex_pred_1, test_size=0.1, shuffle=False)
CPROF_Ex_pred_2_tr, CPROF_Ex_pred_2_tst = train_test_split(CPROF_Ex_pred_2, test_size=0.1, shuffle=False)
CPROF_Ex_pred_3_tr, CPROF_Ex_pred_3_tst = train_test_split(CPROF_Ex_pred_3, test_size=0.1, shuffle=False)
CPROF_Ex_pred_4_tr, CPROF_Ex_pred_4_tst = train_test_split(CPROF_Ex_pred_4, test_size=0.1, shuffle=False)

HOUSING_Ex_pred_1_tr, HOUSING_Ex_pred_1_tst = train_test_split(HOUSING_Ex_pred_1, test_size=0.1, shuffle=False)
HOUSING_Ex_pred_2_tr, HOUSING_Ex_pred_2_tst = train_test_split(HOUSING_Ex_pred_2, test_size=0.1, shuffle=False)
HOUSING_Ex_pred_3_tr, HOUSING_Ex_pred_3_tst = train_test_split(HOUSING_Ex_pred_3, test_size=0.1, shuffle=False)
HOUSING_Ex_pred_4_tr, HOUSING_Ex_pred_4_tst = train_test_split(HOUSING_Ex_pred_4, test_size=0.1, shuffle=False)

INDPROD_Ex_pred_1_tr, INDPROD_Ex_pred_1_tst = train_test_split(INDPROD_Ex_pred_1, test_size=0.1, shuffle=False)
INDPROD_Ex_pred_2_tr, INDPROD_Ex_pred_2_tst = train_test_split(INDPROD_Ex_pred_2, test_size=0.1, shuffle=False)
INDPROD_Ex_pred_3_tr, INDPROD_Ex_pred_3_tst = train_test_split(INDPROD_Ex_pred_3, test_size=0.1, shuffle=False)
INDPROD_Ex_pred_4_tr, INDPROD_Ex_pred_4_tst = train_test_split(INDPROD_Ex_pred_4, test_size=0.1, shuffle=False)

NGDP_Ex_pred_1_tr, NGDP_Ex_pred_1_tst = train_test_split(NGDP_Ex_pred_1, test_size=0.1, shuffle=False)
NGDP_Ex_pred_2_tr, NGDP_Ex_pred_2_tst = train_test_split(NGDP_Ex_pred_2, test_size=0.1, shuffle=False)
NGDP_Ex_pred_3_tr, NGDP_Ex_pred_3_tst = train_test_split(NGDP_Ex_pred_3, test_size=0.1, shuffle=False)
NGDP_Ex_pred_4_tr, NGDP_Ex_pred_4_tst = train_test_split(NGDP_Ex_pred_4, test_size=0.1, shuffle=False)

PGDP_Ex_pred_1_tr, PGDP_Ex_pred_1_tst = train_test_split(PGDP_Ex_pred_1, test_size=0.1, shuffle=False)
PGDP_Ex_pred_2_tr, PGDP_Ex_pred_2_tst = train_test_split(PGDP_Ex_pred_2, test_size=0.1, shuffle=False)
PGDP_Ex_pred_3_tr, PGDP_Ex_pred_3_tst = train_test_split(PGDP_Ex_pred_3, test_size=0.1, shuffle=False)
PGDP_Ex_pred_4_tr, PGDP_Ex_pred_4_tst = train_test_split(PGDP_Ex_pred_4, test_size=0.1, shuffle=False)


#getting the predictions

#%%

#RGDP

np.random.seed(1)

RGDP_nn_pred_1, RGDP_nn_clweights_1 = Base_nn(RGDP_x1_tr, RGDP_Ex_pred_1_tr, RGDP_x1_tst, RGDP_N_exp_1, 'crowd' )
MSE_RGDP_nn_1, MAE_RGDP_nn_1, MSE_RGDP_Mean_pred_1, MAE_RGDP_Mean_pred_1 = model_evaluation(RGDP_nn_pred_1, RGDP_Mean_pred_1_tst,  RGDP_y1_tst)


#%%
RGDP_nn_pred_2, RGDP_nn_clweights_2 = Base_nn(RGDP_x2_tr, RGDP_Ex_pred_2_tr, RGDP_x2_tst, RGDP_N_exp_2, 'crowd')
MSE_RGDP_nn_2, MAE_RGDP_nn_2, MSE_RGDP_Mean_pred_2, MAE_RGDP_Mean_pred_2 = model_evaluation(RGDP_nn_pred_2, RGDP_Mean_pred_2_tst,  RGDP_y2_tst)

RGDP_nn_pred_3, RGDP_nn_clweights_3 = Base_nn(RGDP_x3_tr, RGDP_Ex_pred_3_tr, RGDP_x3_tst, RGDP_N_exp_3, 'crowd')
MSE_RGDP_nn_3, MAE_RGDP_nn_3, MSE_RGDP_Mean_pred_3, MAE_RGDP_Mean_pred_3 = model_evaluation(RGDP_nn_pred_3, RGDP_Mean_pred_3_tst,  RGDP_y3_tst)

RGDP_nn_pred_4, RGDP_nn_clweights_4 = Base_nn(RGDP_x4_tr, RGDP_Ex_pred_4_tr, RGDP_x4_tst, RGDP_N_exp_4, 'crowd')
MSE_RGDP_nn_4, MAE_RGDP_nn_4, MSE_RGDP_Mean_pred_4, MAE_RGDP_Mean_pred_4 = model_evaluation(RGDP_nn_pred_4, RGDP_Mean_pred_4_tst,  RGDP_y4_tst)



#%%
np.random.seed(1)

UNEMP_nn_pred_1, UNEMP_nn_clweights_1 = Base_nn(UNEMP_x1_tr, UNEMP_Ex_pred_1_tr, UNEMP_x1_tst, UNEMP_N_exp_1, 'crowd' )
MSE_UNEMP_nn_1, MAE_UNEMP_nn_1, MSE_UNEMP_Mean_pred_1, MAE_UNEMP_Mean_pred_1 = model_evaluation(UNEMP_nn_pred_1, UNEMP_Mean_pred_1_tst,  UNEMP_y1_tst)

UNEMP_nn_pred_2, UNEMP_nn_clweights_2 = Base_nn(UNEMP_x2_tr, UNEMP_Ex_pred_2_tr, UNEMP_x2_tst, UNEMP_N_exp_2, 'crowd')
MSE_UNEMP_nn_2, MAE_UNEMP_nn_2, MSE_UNEMP_Mean_pred_2, MAE_UNEMP_Mean_pred_2 = model_evaluation(UNEMP_nn_pred_2, UNEMP_Mean_pred_2_tst,  UNEMP_y2_tst)

UNEMP_nn_pred_3, UNEMP_nn_clweights_3 = Base_nn(UNEMP_x3_tr, UNEMP_Ex_pred_3_tr, UNEMP_x3_tst, UNEMP_N_exp_3, 'crowd')
MSE_UNEMP_nn_3, MAE_UNEMP_nn_3, MSE_UNEMP_Mean_pred_3, MAE_UNEMP_Mean_pred_3 = model_evaluation(UNEMP_nn_pred_3, UNEMP_Mean_pred_3_tst,  UNEMP_y3_tst)

UNEMP_nn_pred_4, UNEMP_nn_clweights_4 = Base_nn(UNEMP_x4_tr, UNEMP_Ex_pred_4_tr, UNEMP_x4_tst, UNEMP_N_exp_4, 'crowd')
MSE_UNEMP_nn_4, MAE_UNEMP_nn_4, MSE_UNEMP_Mean_pred_4, MAE_UNEMP_Mean_pred_4 = model_evaluation(UNEMP_nn_pred_4, UNEMP_Mean_pred_4_tst,  UNEMP_y4_tst)

#%%
np.random.seed(1)

CPROF_nn_pred_1, CPROF_nn_clweights_1 = Base_nn(CPROF_x1_tr, CPROF_Ex_pred_1_tr, CPROF_x1_tst, CPROF_N_exp_1, 'crowd' )
MSE_CPROF_nn_1, MAE_CPROF_nn_1, MSE_CPROF_Mean_pred_1, MAE_CPROF_Mean_pred_1 = model_evaluation(CPROF_nn_pred_1, CPROF_Mean_pred_1_tst,  CPROF_y1_tst)

CPROF_nn_pred_2, CPROF_nn_clweights_2 = Base_nn(CPROF_x2_tr, CPROF_Ex_pred_2_tr, CPROF_x2_tst, CPROF_N_exp_2, 'crowd')
MSE_CPROF_nn_2, MAE_CPROF_nn_2, MSE_CPROF_Mean_pred_2, MAE_CPROF_Mean_pred_2 = model_evaluation(CPROF_nn_pred_2, CPROF_Mean_pred_2_tst,  CPROF_y2_tst)

CPROF_nn_pred_3, CPROF_nn_clweights_3 = Base_nn(CPROF_x3_tr, CPROF_Ex_pred_3_tr, CPROF_x3_tst, CPROF_N_exp_3, 'crowd')
MSE_CPROF_nn_3, MAE_CPROF_nn_3, MSE_CPROF_Mean_pred_3, MAE_CPROF_Mean_pred_3 = model_evaluation(CPROF_nn_pred_3, CPROF_Mean_pred_3_tst,  CPROF_y3_tst)

CPROF_nn_pred_4, CPROF_nn_clweights_4 = Base_nn(CPROF_x4_tr, CPROF_Ex_pred_4_tr, CPROF_x4_tst, CPROF_N_exp_4, 'crowd')
MSE_CPROF_nn_4, MAE_CPROF_nn_4, MSE_CPROF_Mean_pred_4, MAE_CPROF_Mean_pred_4 = model_evaluation(CPROF_nn_pred_4, CPROF_Mean_pred_4_tst,  CPROF_y4_tst)

#%%
np.random.seed(1)

HOUSING_nn_pred_1, HOUSING_nn_clweights_1 = Base_nn(HOUSING_x1_tr, HOUSING_Ex_pred_1_tr, HOUSING_x1_tst, HOUSING_N_exp_1, 'crowd' )
MSE_HOUSING_nn_1, MAE_HOUSING_nn_1, MSE_HOUSING_Mean_pred_1, MAE_HOUSING_Mean_pred_1 = model_evaluation(HOUSING_nn_pred_1, HOUSING_Mean_pred_1_tst,  HOUSING_y1_tst)

HOUSING_nn_pred_2, HOUSING_nn_clweights_2 = Base_nn(HOUSING_x2_tr, HOUSING_Ex_pred_2_tr, HOUSING_x2_tst, HOUSING_N_exp_2, 'crowd')
MSE_HOUSING_nn_2, MAE_HOUSING_nn_2, MSE_HOUSING_Mean_pred_2, MAE_HOUSING_Mean_pred_2 = model_evaluation(HOUSING_nn_pred_2, HOUSING_Mean_pred_2_tst,  HOUSING_y2_tst)

HOUSING_nn_pred_3, HOUSING_nn_clweights_3 = Base_nn(HOUSING_x3_tr, HOUSING_Ex_pred_3_tr, HOUSING_x3_tst, HOUSING_N_exp_3, 'crowd')
MSE_HOUSING_nn_3, MAE_HOUSING_nn_3, MSE_HOUSING_Mean_pred_3, MAE_HOUSING_Mean_pred_3 = model_evaluation(HOUSING_nn_pred_3, HOUSING_Mean_pred_3_tst,  HOUSING_y3_tst)

HOUSING_nn_pred_4, HOUSING_nn_clweights_4 = Base_nn(HOUSING_x4_tr, HOUSING_Ex_pred_4_tr, HOUSING_x4_tst, HOUSING_N_exp_4, 'crowd')
MSE_HOUSING_nn_4, MAE_HOUSING_nn_4, MSE_HOUSING_Mean_pred_4, MAE_HOUSING_Mean_pred_4 = model_evaluation(HOUSING_nn_pred_4, HOUSING_Mean_pred_4_tst,  HOUSING_y4_tst)

#%%
np.random.seed(1)

INDPROD_nn_pred_1, INDPROD_nn_clweights_1 = Base_nn(INDPROD_x1_tr, INDPROD_Ex_pred_1_tr, INDPROD_x1_tst, INDPROD_N_exp_1, 'crowd' )
MSE_INDPROD_nn_1, MAE_INDPROD_nn_1, MSE_INDPROD_Mean_pred_1, MAE_INDPROD_Mean_pred_1 = model_evaluation(INDPROD_nn_pred_1, INDPROD_Mean_pred_1_tst,  INDPROD_y1_tst)

INDPROD_nn_pred_2, INDPROD_nn_clweights_2 = Base_nn(INDPROD_x2_tr, INDPROD_Ex_pred_2_tr, INDPROD_x2_tst, INDPROD_N_exp_2, 'crowd')
MSE_INDPROD_nn_2, MAE_INDPROD_nn_2, MSE_INDPROD_Mean_pred_2, MAE_INDPROD_Mean_pred_2 = model_evaluation(INDPROD_nn_pred_2, INDPROD_Mean_pred_2_tst,  INDPROD_y2_tst)

INDPROD_nn_pred_3, INDPROD_nn_clweights_3 = Base_nn(INDPROD_x3_tr, INDPROD_Ex_pred_3_tr, INDPROD_x3_tst, INDPROD_N_exp_3, 'crowd')
MSE_INDPROD_nn_3, MAE_INDPROD_nn_3, MSE_INDPROD_Mean_pred_3, MAE_INDPROD_Mean_pred_3 = model_evaluation(INDPROD_nn_pred_3, INDPROD_Mean_pred_3_tst,  INDPROD_y3_tst)

INDPROD_nn_pred_4, INDPROD_nn_clweights_4 = Base_nn(INDPROD_x4_tr, INDPROD_Ex_pred_4_tr, INDPROD_x4_tst, INDPROD_N_exp_4, 'crowd')
MSE_INDPROD_nn_4, MAE_INDPROD_nn_4, MSE_INDPROD_Mean_pred_4, MAE_INDPROD_Mean_pred_4 = model_evaluation(INDPROD_nn_pred_4, INDPROD_Mean_pred_4_tst,  INDPROD_y4_tst)

#%%
np.random.seed(1)

NGDP_nn_pred_1, NGDP_nn_clweights_1 = Base_nn(NGDP_x1_tr, NGDP_Ex_pred_1_tr, NGDP_x1_tst, NGDP_N_exp_1, 'crowd' )
MSE_NGDP_nn_1, MAE_NGDP_nn_1, MSE_NGDP_Mean_pred_1, MAE_NGDP_Mean_pred_1 = model_evaluation(NGDP_nn_pred_1, NGDP_Mean_pred_1_tst,  NGDP_y1_tst)

NGDP_nn_pred_2, NGDP_nn_clweights_2 = Base_nn(NGDP_x2_tr, NGDP_Ex_pred_2_tr, NGDP_x2_tst, NGDP_N_exp_2, 'crowd')
MSE_NGDP_nn_2, MAE_NGDP_nn_2, MSE_NGDP_Mean_pred_2, MAE_NGDP_Mean_pred_2 = model_evaluation(NGDP_nn_pred_2, NGDP_Mean_pred_2_tst,  NGDP_y2_tst)

NGDP_nn_pred_3, NGDP_nn_clweights_3 = Base_nn(NGDP_x3_tr, NGDP_Ex_pred_3_tr, NGDP_x3_tst, NGDP_N_exp_3, 'crowd')
MSE_NGDP_nn_3, MAE_NGDP_nn_3, MSE_NGDP_Mean_pred_3, MAE_NGDP_Mean_pred_3 = model_evaluation(NGDP_nn_pred_3, NGDP_Mean_pred_3_tst,  NGDP_y3_tst)

NGDP_nn_pred_4, NGDP_nn_clweights_4 = Base_nn(NGDP_x4_tr, NGDP_Ex_pred_4_tr, NGDP_x4_tst, NGDP_N_exp_4, 'crowd')
MSE_NGDP_nn_4, MAE_NGDP_nn_4, MSE_NGDP_Mean_pred_4, MAE_NGDP_Mean_pred_4 = model_evaluation(NGDP_nn_pred_4, NGDP_Mean_pred_4_tst,  NGDP_y4_tst)

#%%
np.random.seed(1)

PGDP_nn_pred_1, PGDP_nn_clweights_1 = Base_nn(PGDP_x1_tr, PGDP_Ex_pred_1_tr, PGDP_x1_tst, PGDP_N_exp_1, 'crowd' )
MSE_PGDP_nn_1, MAE_PGDP_nn_1, MSE_PGDP_Mean_pred_1, MAE_PGDP_Mean_pred_1 = model_evaluation(PGDP_nn_pred_1, PGDP_Mean_pred_1_tst,  PGDP_y1_tst)

PGDP_nn_pred_2, PGDP_nn_clweights_2 = Base_nn(PGDP_x2_tr, PGDP_Ex_pred_2_tr, PGDP_x2_tst, PGDP_N_exp_2, 'crowd')
MSE_PGDP_nn_2, MAE_PGDP_nn_2, MSE_PGDP_Mean_pred_2, MAE_PGDP_Mean_pred_2 = model_evaluation(PGDP_nn_pred_2, PGDP_Mean_pred_2_tst,  PGDP_y2_tst)

PGDP_nn_pred_3, PGDP_nn_clweights_3 = Base_nn(PGDP_x3_tr, PGDP_Ex_pred_3_tr, PGDP_x3_tst, PGDP_N_exp_3, 'crowd')
MSE_PGDP_nn_3, MAE_PGDP_nn_3, MSE_PGDP_Mean_pred_3, MAE_PGDP_Mean_pred_3 = model_evaluation(PGDP_nn_pred_3, PGDP_Mean_pred_3_tst,  PGDP_y3_tst)

PGDP_nn_pred_4, PGDP_nn_clweights_4 = Base_nn(PGDP_x4_tr, PGDP_Ex_pred_4_tr, PGDP_x4_tst, PGDP_N_exp_4, 'crowd')
MSE_PGDP_nn_4, MAE_PGDP_nn_4, MSE_PGDP_Mean_pred_4, MAE_PGDP_Mean_pred_4 = model_evaluation(PGDP_nn_pred_4, PGDP_Mean_pred_4_tst,  PGDP_y4_tst)




#%%

#neural networks with ground truth targets

np.random.seed(1)

RGDP_nn_t_pred_1, RGDP_nn_t_clweights_1 = Base_nn(RGDP_x1_tr, RGDP_y1_tr, RGDP_x1_tst, RGDP_N_exp_1, 'neh' )
MSE_RGDP_nn_t_1, MAE_RGDP_nn_t_1, MSE_RGDP_Mean_pred_1, MAE_RGDP_Mean_pred_1 = model_evaluation(RGDP_nn_t_pred_1, RGDP_Mean_pred_1_tst,  RGDP_y1_tst)

RGDP_nn_t_pred_2, RGDP_nn_t_clweights_2 = Base_nn(RGDP_x2_tr, RGDP_y2_tr, RGDP_x2_tst, RGDP_N_exp_2, 'neh')
MSE_RGDP_nn_t_2, MAE_RGDP_nn_t_2, MSE_RGDP_Mean_pred_2, MAE_RGDP_Mean_pred_2 = model_evaluation(RGDP_nn_t_pred_2, RGDP_Mean_pred_2_tst,  RGDP_y2_tst)

RGDP_nn_t_pred_3, RGDP_nn_t_clweights_3 = Base_nn(RGDP_x3_tr, RGDP_y3_tr, RGDP_x3_tst, RGDP_N_exp_3, 'neh')
MSE_RGDP_nn_t_3, MAE_RGDP_nn_t_3, MSE_RGDP_Mean_pred_3, MAE_RGDP_Mean_pred_3 = model_evaluation(RGDP_nn_t_pred_3, RGDP_Mean_pred_3_tst,  RGDP_y3_tst)

RGDP_nn_t_pred_4, RGDP_nn_t_clweights_4 = Base_nn(RGDP_x4_tr, RGDP_y4_tr, RGDP_x4_tst, RGDP_N_exp_4, 'neh')
MSE_RGDP_nn_t_4, MAE_RGDP_nn_t_4, MSE_RGDP_Mean_pred_4, MAE_RGDP_Mean_pred_4 = model_evaluation(RGDP_nn_t_pred_4, RGDP_Mean_pred_4_tst,  RGDP_y4_tst)



#%%
np.random.seed(1)

UNEMP_nn_t_pred_1, UNEMP_nn_t_clweights_1 = Base_nn(UNEMP_x1_tr, UNEMP_y1_tr, UNEMP_x1_tst, UNEMP_N_exp_1, 'neh' )
MSE_UNEMP_nn_t_1, MAE_UNEMP_nn_t_1, MSE_UNEMP_Mean_pred_1, MAE_UNEMP_Mean_pred_1 = model_evaluation(UNEMP_nn_t_pred_1, UNEMP_Mean_pred_1_tst,  UNEMP_y1_tst)

UNEMP_nn_t_pred_2, UNEMP_nn_t_clweights_2 = Base_nn(UNEMP_x2_tr, UNEMP_y2_tr, UNEMP_x2_tst, UNEMP_N_exp_2, 'neh')
MSE_UNEMP_nn_t_2, MAE_UNEMP_nn_t_2, MSE_UNEMP_Mean_pred_2, MAE_UNEMP_Mean_pred_2 = model_evaluation(UNEMP_nn_t_pred_2, UNEMP_Mean_pred_2_tst,  UNEMP_y2_tst)

UNEMP_nn_t_pred_3, UNEMP_nn_t_clweights_3 = Base_nn(UNEMP_x3_tr, UNEMP_y3_tr, UNEMP_x3_tst, UNEMP_N_exp_3, 'neh')
MSE_UNEMP_nn_t_3, MAE_UNEMP_nn_t_3, MSE_UNEMP_Mean_pred_3, MAE_UNEMP_Mean_pred_3 = model_evaluation(UNEMP_nn_t_pred_3, UNEMP_Mean_pred_3_tst,  UNEMP_y3_tst)

UNEMP_nn_t_pred_4, UNEMP_nn_t_clweights_4 = Base_nn(UNEMP_x4_tr, UNEMP_y4_tr, UNEMP_x4_tst, UNEMP_N_exp_4, 'neh')
MSE_UNEMP_nn_t_4, MAE_UNEMP_nn_t_4, MSE_UNEMP_Mean_pred_4, MAE_UNEMP_Mean_pred_4 = model_evaluation(UNEMP_nn_t_pred_4, UNEMP_Mean_pred_4_tst,  UNEMP_y4_tst)

#%%
np.random.seed(1)

CPROF_nn_t_pred_1, CPROF_nn_t_clweights_1 = Base_nn(CPROF_x1_tr, CPROF_y1_tr, CPROF_x1_tst, CPROF_N_exp_1, 'neh' )
MSE_CPROF_nn_t_1, MAE_CPROF_nn_t_1, MSE_CPROF_Mean_pred_1, MAE_CPROF_Mean_pred_1 = model_evaluation(CPROF_nn_t_pred_1, CPROF_Mean_pred_1_tst,  CPROF_y1_tst)

CPROF_nn_t_pred_2, CPROF_nn_t_clweights_2 = Base_nn(CPROF_x2_tr, CPROF_y2_tr, CPROF_x2_tst, CPROF_N_exp_2, 'neh')
MSE_CPROF_nn_t_2, MAE_CPROF_nn_t_2, MSE_CPROF_Mean_pred_2, MAE_CPROF_Mean_pred_2 = model_evaluation(CPROF_nn_t_pred_2, CPROF_Mean_pred_2_tst,  CPROF_y2_tst)

CPROF_nn_t_pred_3, CPROF_nn_t_clweights_3 = Base_nn(CPROF_x3_tr, CPROF_y3_tr, CPROF_x3_tst, CPROF_N_exp_3, 'neh')
MSE_CPROF_nn_t_3, MAE_CPROF_nn_t_3, MSE_CPROF_Mean_pred_3, MAE_CPROF_Mean_pred_3 = model_evaluation(CPROF_nn_t_pred_3, CPROF_Mean_pred_3_tst,  CPROF_y3_tst)

CPROF_nn_t_pred_4, CPROF_nn_t_clweights_4 = Base_nn(CPROF_x4_tr, CPROF_y4_tr, CPROF_x4_tst, CPROF_N_exp_4, 'neh')
MSE_CPROF_nn_t_4, MAE_CPROF_nn_t_4, MSE_CPROF_Mean_pred_4, MAE_CPROF_Mean_pred_4 = model_evaluation(CPROF_nn_t_pred_4, CPROF_Mean_pred_4_tst,  CPROF_y4_tst)

#%%
np.random.seed(1)

HOUSING_nn_t_pred_1, HOUSING_nn_t_clweights_1 = Base_nn(HOUSING_x1_tr, HOUSING_y1_tr, HOUSING_x1_tst, HOUSING_N_exp_1, 'neh' )
MSE_HOUSING_nn_t_1, MAE_HOUSING_nn_t_1, MSE_HOUSING_Mean_pred_1, MAE_HOUSING_Mean_pred_1 = model_evaluation(HOUSING_nn_t_pred_1, HOUSING_Mean_pred_1_tst,  HOUSING_y1_tst)

HOUSING_nn_t_pred_2, HOUSING_nn_t_clweights_2 = Base_nn(HOUSING_x2_tr, HOUSING_y2_tr, HOUSING_x2_tst, HOUSING_N_exp_2, 'neh')
MSE_HOUSING_nn_t_2, MAE_HOUSING_nn_t_2, MSE_HOUSING_Mean_pred_2, MAE_HOUSING_Mean_pred_2 = model_evaluation(HOUSING_nn_t_pred_2, HOUSING_Mean_pred_2_tst,  HOUSING_y2_tst)

HOUSING_nn_t_pred_3, HOUSING_nn_t_clweights_3 = Base_nn(HOUSING_x3_tr, HOUSING_y3_tr, HOUSING_x3_tst, HOUSING_N_exp_3, 'neh')
MSE_HOUSING_nn_t_3, MAE_HOUSING_nn_t_3, MSE_HOUSING_Mean_pred_3, MAE_HOUSING_Mean_pred_3 = model_evaluation(HOUSING_nn_t_pred_3, HOUSING_Mean_pred_3_tst,  HOUSING_y3_tst)

HOUSING_nn_t_pred_4, HOUSING_nn_t_clweights_4 = Base_nn(HOUSING_x4_tr, HOUSING_y4_tr, HOUSING_x4_tst, HOUSING_N_exp_4, 'neh')
MSE_HOUSING_nn_t_4, MAE_HOUSING_nn_t_4, MSE_HOUSING_Mean_pred_4, MAE_HOUSING_Mean_pred_4 = model_evaluation(HOUSING_nn_t_pred_4, HOUSING_Mean_pred_4_tst,  HOUSING_y4_tst)

#%%
np.random.seed(1)

INDPROD_nn_t_pred_1, INDPROD_nn_t_clweights_1 = Base_nn(INDPROD_x1_tr, INDPROD_y1_tr, INDPROD_x1_tst, INDPROD_N_exp_1, 'neh' )
MSE_INDPROD_nn_t_1, MAE_INDPROD_nn_t_1, MSE_INDPROD_Mean_pred_1, MAE_INDPROD_Mean_pred_1 = model_evaluation(INDPROD_nn_t_pred_1, INDPROD_Mean_pred_1_tst,  INDPROD_y1_tst)

INDPROD_nn_t_pred_2, INDPROD_nn_t_clweights_2 = Base_nn(INDPROD_x2_tr, INDPROD_y2_tr, INDPROD_x2_tst, INDPROD_N_exp_2, 'neh')
MSE_INDPROD_nn_t_2, MAE_INDPROD_nn_t_2, MSE_INDPROD_Mean_pred_2, MAE_INDPROD_Mean_pred_2 = model_evaluation(INDPROD_nn_t_pred_2, INDPROD_Mean_pred_2_tst,  INDPROD_y2_tst)

INDPROD_nn_t_pred_3, INDPROD_nn_t_clweights_3 = Base_nn(INDPROD_x3_tr, INDPROD_y3_tr, INDPROD_x3_tst, INDPROD_N_exp_3, 'neh')
MSE_INDPROD_nn_t_3, MAE_INDPROD_nn_t_3, MSE_INDPROD_Mean_pred_3, MAE_INDPROD_Mean_pred_3 = model_evaluation(INDPROD_nn_t_pred_3, INDPROD_Mean_pred_3_tst,  INDPROD_y3_tst)

INDPROD_nn_t_pred_4, INDPROD_nn_t_clweights_4 = Base_nn(INDPROD_x4_tr, INDPROD_y4_tr, INDPROD_x4_tst, INDPROD_N_exp_4, 'neh')
MSE_INDPROD_nn_t_4, MAE_INDPROD_nn_t_4, MSE_INDPROD_Mean_pred_4, MAE_INDPROD_Mean_pred_4 = model_evaluation(INDPROD_nn_t_pred_4, INDPROD_Mean_pred_4_tst,  INDPROD_y4_tst)

#%%
np.random.seed(1)

NGDP_nn_t_pred_1, NGDP_nn_t_clweights_1 = Base_nn(NGDP_x1_tr, NGDP_y1_tr, NGDP_x1_tst, NGDP_N_exp_1, 'neh' )
MSE_NGDP_nn_t_1, MAE_NGDP_nn_t_1, MSE_NGDP_Mean_pred_1, MAE_NGDP_Mean_pred_1 = model_evaluation(NGDP_nn_t_pred_1, NGDP_Mean_pred_1_tst,  NGDP_y1_tst)

NGDP_nn_t_pred_2, NGDP_nn_t_clweights_2 = Base_nn(NGDP_x2_tr, NGDP_y2_tr, NGDP_x2_tst, NGDP_N_exp_2, 'neh')
MSE_NGDP_nn_t_2, MAE_NGDP_nn_t_2, MSE_NGDP_Mean_pred_2, MAE_NGDP_Mean_pred_2 = model_evaluation(NGDP_nn_t_pred_2, NGDP_Mean_pred_2_tst,  NGDP_y2_tst)

NGDP_nn_t_pred_3, NGDP_nn_t_clweights_3 = Base_nn(NGDP_x3_tr, NGDP_y3_tr, NGDP_x3_tst, NGDP_N_exp_3, 'neh')
MSE_NGDP_nn_t_3, MAE_NGDP_nn_t_3, MSE_NGDP_Mean_pred_3, MAE_NGDP_Mean_pred_3 = model_evaluation(NGDP_nn_t_pred_3, NGDP_Mean_pred_3_tst,  NGDP_y3_tst)

NGDP_nn_t_pred_4, NGDP_nn_t_clweights_4 = Base_nn(NGDP_x4_tr, NGDP_y4_tr, NGDP_x4_tst, NGDP_N_exp_4, 'neh')
MSE_NGDP_nn_t_4, MAE_NGDP_nn_t_4, MSE_NGDP_Mean_pred_4, MAE_NGDP_Mean_pred_4 = model_evaluation(NGDP_nn_t_pred_4, NGDP_Mean_pred_4_tst,  NGDP_y4_tst)

#%%
np.random.seed(1)

PGDP_nn_t_pred_1, PGDP_nn_t_clweights_1 = Base_nn(PGDP_x1_tr, PGDP_y1_tr, PGDP_x1_tst, PGDP_N_exp_1, 'neh' )
MSE_PGDP_nn_t_1, MAE_PGDP_nn_t_1, MSE_PGDP_Mean_pred_1, MAE_PGDP_Mean_pred_1 = model_evaluation(PGDP_nn_t_pred_1, PGDP_Mean_pred_1_tst,  PGDP_y1_tst)

PGDP_nn_t_pred_2, PGDP_nn_t_clweights_2 = Base_nn(PGDP_x2_tr, PGDP_y2_tr, PGDP_x2_tst, PGDP_N_exp_2, 'neh')
MSE_PGDP_nn_t_2, MAE_PGDP_nn_t_2, MSE_PGDP_Mean_pred_2, MAE_PGDP_Mean_pred_2 = model_evaluation(PGDP_nn_t_pred_2, PGDP_Mean_pred_2_tst,  PGDP_y2_tst)

PGDP_nn_t_pred_3, PGDP_nn_t_clweights_3 = Base_nn(PGDP_x3_tr, PGDP_y3_tr, PGDP_x3_tst, PGDP_N_exp_3, 'neh')
MSE_PGDP_nn_t_3, MAE_PGDP_nn_t_3, MSE_PGDP_Mean_pred_3, MAE_PGDP_Mean_pred_3 = model_evaluation(PGDP_nn_t_pred_3, PGDP_Mean_pred_3_tst,  PGDP_y3_tst)

PGDP_nn_t_pred_4, PGDP_nn_t_clweights_4 = Base_nn(PGDP_x4_tr, PGDP_y4_tr, PGDP_x4_tst, PGDP_N_exp_4, 'neh')
MSE_PGDP_nn_t_4, MAE_PGDP_nn_t_4, MSE_PGDP_Mean_pred_4, MAE_PGDP_Mean_pred_4 = model_evaluation(PGDP_nn_t_pred_4, PGDP_Mean_pred_4_tst,  PGDP_y4_tst)












#%%
#plotting




plot_preds(RGDP_y1_tst, RGDP_Mean_pred_1_tst, RGDP_nn_pred_1, RGDP_nn_t_pred_1  )
print_error(MSE_RGDP_nn_1, MAE_RGDP_nn_1, MSE_RGDP_Mean_pred_1, MAE_RGDP_Mean_pred_1, 'RGDP_1', MSE_RGDP_nn_t_1, MAE_RGDP_nn_t_1  )
plot_preds(RGDP_y2_tst, RGDP_Mean_pred_2_tst, RGDP_nn_pred_2, RGDP_nn_t_pred_2 )
print_error(MSE_RGDP_nn_2, MAE_RGDP_nn_2, MSE_RGDP_Mean_pred_2, MAE_RGDP_Mean_pred_2, 'RGDP_2', MSE_RGDP_nn_t_2, MAE_RGDP_nn_t_2  )
plot_preds(RGDP_y3_tst, RGDP_Mean_pred_3_tst, RGDP_nn_pred_3, RGDP_nn_t_pred_3 )
print_error(MSE_RGDP_nn_3, MAE_RGDP_nn_3, MSE_RGDP_Mean_pred_3, MAE_RGDP_Mean_pred_3, 'RGDP_3', MSE_RGDP_nn_t_3, MAE_RGDP_nn_t_3  )
plot_preds(RGDP_y4_tst, RGDP_Mean_pred_4_tst, RGDP_nn_pred_4, RGDP_nn_t_pred_4 )
print_error(MSE_RGDP_nn_4, MAE_RGDP_nn_4, MSE_RGDP_Mean_pred_4, MAE_RGDP_Mean_pred_4, 'RGDP_4', MSE_RGDP_nn_t_4, MAE_RGDP_nn_t_4  )
#%%

plot_preds(UNEMP_y1_tst, UNEMP_Mean_pred_1_tst, UNEMP_nn_pred_1, UNEMP_nn_t_pred_1  )
print_error(MSE_UNEMP_nn_1, MAE_UNEMP_nn_1, MSE_UNEMP_Mean_pred_1, MAE_UNEMP_Mean_pred_1, 'UNEMP_1', MSE_UNEMP_nn_t_1, MAE_UNEMP_nn_t_1  )
plot_preds(UNEMP_y2_tst, UNEMP_Mean_pred_2_tst, UNEMP_nn_pred_2, UNEMP_nn_t_pred_2 )
print_error(MSE_UNEMP_nn_2, MAE_UNEMP_nn_2, MSE_UNEMP_Mean_pred_2, MAE_UNEMP_Mean_pred_2, 'UNEMP_2', MSE_UNEMP_nn_t_2, MAE_UNEMP_nn_t_2  )
plot_preds(UNEMP_y3_tst, UNEMP_Mean_pred_3_tst, UNEMP_nn_pred_3, UNEMP_nn_t_pred_3 )
print_error(MSE_UNEMP_nn_3, MAE_UNEMP_nn_3, MSE_UNEMP_Mean_pred_3, MAE_UNEMP_Mean_pred_3, 'UNEMP_3', MSE_UNEMP_nn_t_3, MAE_UNEMP_nn_t_3  )
plot_preds(UNEMP_y4_tst, UNEMP_Mean_pred_4_tst, UNEMP_nn_pred_4, UNEMP_nn_t_pred_4 )
print_error(MSE_UNEMP_nn_4, MAE_UNEMP_nn_4, MSE_UNEMP_Mean_pred_4, MAE_UNEMP_Mean_pred_4, 'UNEMP_4', MSE_UNEMP_nn_t_4, MAE_UNEMP_nn_t_4  )
#%%

plot_preds(CPROF_y1_tst, CPROF_Mean_pred_1_tst, CPROF_nn_pred_1, CPROF_nn_t_pred_1  )
print_error(MSE_CPROF_nn_1, MAE_CPROF_nn_1, MSE_CPROF_Mean_pred_1, MAE_CPROF_Mean_pred_1, 'CPROF_1', MSE_CPROF_nn_t_1, MAE_CPROF_nn_t_1  )
plot_preds(CPROF_y2_tst, CPROF_Mean_pred_2_tst, CPROF_nn_pred_2, CPROF_nn_t_pred_2 )
print_error(MSE_CPROF_nn_2, MAE_CPROF_nn_2, MSE_CPROF_Mean_pred_2, MAE_CPROF_Mean_pred_2, 'CPROF_2', MSE_CPROF_nn_t_2, MAE_CPROF_nn_t_2  )
plot_preds(CPROF_y3_tst, CPROF_Mean_pred_3_tst, CPROF_nn_pred_3, CPROF_nn_t_pred_3 )
print_error(MSE_CPROF_nn_3, MAE_CPROF_nn_3, MSE_CPROF_Mean_pred_3, MAE_CPROF_Mean_pred_3, 'CPROF_3', MSE_CPROF_nn_t_3, MAE_CPROF_nn_t_3  )
plot_preds(CPROF_y4_tst, CPROF_Mean_pred_4_tst, CPROF_nn_pred_4, CPROF_nn_t_pred_4 )
print_error(MSE_CPROF_nn_4, MAE_CPROF_nn_4, MSE_CPROF_Mean_pred_4, MAE_CPROF_Mean_pred_4, 'CPROF_4', MSE_CPROF_nn_t_4, MAE_CPROF_nn_t_4  )
#%%

plot_preds(HOUSING_y1_tst, HOUSING_Mean_pred_1_tst, HOUSING_nn_pred_1, HOUSING_nn_t_pred_1  )
print_error(MSE_HOUSING_nn_1, MAE_HOUSING_nn_1, MSE_HOUSING_Mean_pred_1, MAE_HOUSING_Mean_pred_1, 'HOUSING_1', MSE_HOUSING_nn_t_1, MAE_HOUSING_nn_t_1  )
plot_preds(HOUSING_y2_tst, HOUSING_Mean_pred_2_tst, HOUSING_nn_pred_2, HOUSING_nn_t_pred_2 )
print_error(MSE_HOUSING_nn_2, MAE_HOUSING_nn_2, MSE_HOUSING_Mean_pred_2, MAE_HOUSING_Mean_pred_2, 'HOUSING_2', MSE_HOUSING_nn_t_2, MAE_HOUSING_nn_t_2  )
plot_preds(HOUSING_y3_tst, HOUSING_Mean_pred_3_tst, HOUSING_nn_pred_3, HOUSING_nn_t_pred_3 )
print_error(MSE_HOUSING_nn_3, MAE_HOUSING_nn_3, MSE_HOUSING_Mean_pred_3, MAE_HOUSING_Mean_pred_3, 'HOUSING_3', MSE_HOUSING_nn_t_3, MAE_HOUSING_nn_t_3  )
plot_preds(HOUSING_y4_tst, HOUSING_Mean_pred_4_tst, HOUSING_nn_pred_4, HOUSING_nn_t_pred_4 )
print_error(MSE_HOUSING_nn_4, MAE_HOUSING_nn_4, MSE_HOUSING_Mean_pred_4, MAE_HOUSING_Mean_pred_4, 'HOUSING_4', MSE_HOUSING_nn_t_4, MAE_HOUSING_nn_t_4  )
#%%

plot_preds(INDPROD_y1_tst, INDPROD_Mean_pred_1_tst, INDPROD_nn_pred_1, INDPROD_nn_t_pred_1  )
print_error(MSE_INDPROD_nn_1, MAE_INDPROD_nn_1, MSE_INDPROD_Mean_pred_1, MAE_INDPROD_Mean_pred_1, 'INDPROD_1', MSE_INDPROD_nn_t_1, MAE_INDPROD_nn_t_1  )
plot_preds(INDPROD_y2_tst, INDPROD_Mean_pred_2_tst, INDPROD_nn_pred_2, INDPROD_nn_t_pred_2 )
print_error(MSE_INDPROD_nn_2, MAE_INDPROD_nn_2, MSE_INDPROD_Mean_pred_2, MAE_INDPROD_Mean_pred_2, 'INDPROD_2', MSE_INDPROD_nn_t_2, MAE_INDPROD_nn_t_2  )
plot_preds(INDPROD_y3_tst, INDPROD_Mean_pred_3_tst, INDPROD_nn_pred_3, INDPROD_nn_t_pred_3 )
print_error(MSE_INDPROD_nn_3, MAE_INDPROD_nn_3, MSE_INDPROD_Mean_pred_3, MAE_INDPROD_Mean_pred_3, 'INDPROD_3', MSE_INDPROD_nn_t_3, MAE_INDPROD_nn_t_3  )
plot_preds(INDPROD_y4_tst, INDPROD_Mean_pred_4_tst, INDPROD_nn_pred_4, INDPROD_nn_t_pred_4 )
print_error(MSE_INDPROD_nn_4, MAE_INDPROD_nn_4, MSE_INDPROD_Mean_pred_4, MAE_INDPROD_Mean_pred_4, 'INDPROD_4', MSE_INDPROD_nn_t_4, MAE_INDPROD_nn_t_4  )
#%%

plot_preds(NGDP_y1_tst, NGDP_Mean_pred_1_tst, NGDP_nn_pred_1, NGDP_nn_t_pred_1  )
print_error(MSE_NGDP_nn_1, MAE_NGDP_nn_1, MSE_NGDP_Mean_pred_1, MAE_NGDP_Mean_pred_1, 'NGDP_1', MSE_NGDP_nn_t_1, MAE_NGDP_nn_t_1  )
plot_preds(NGDP_y2_tst, NGDP_Mean_pred_2_tst, NGDP_nn_pred_2, NGDP_nn_t_pred_2 )
print_error(MSE_NGDP_nn_2, MAE_NGDP_nn_2, MSE_NGDP_Mean_pred_2, MAE_NGDP_Mean_pred_2, 'NGDP_2', MSE_NGDP_nn_t_2, MAE_NGDP_nn_t_2  )
plot_preds(NGDP_y3_tst, NGDP_Mean_pred_3_tst, NGDP_nn_pred_3, NGDP_nn_t_pred_3 )
print_error(MSE_NGDP_nn_3, MAE_NGDP_nn_3, MSE_NGDP_Mean_pred_3, MAE_NGDP_Mean_pred_3, 'NGDP_3', MSE_NGDP_nn_t_3, MAE_NGDP_nn_t_3  )
plot_preds(NGDP_y4_tst, NGDP_Mean_pred_4_tst, NGDP_nn_pred_4, NGDP_nn_t_pred_4 )
print_error(MSE_NGDP_nn_4, MAE_NGDP_nn_4, MSE_NGDP_Mean_pred_4, MAE_NGDP_Mean_pred_4, 'NGDP_4', MSE_NGDP_nn_t_4, MAE_NGDP_nn_t_4  )
#%%

plot_preds(PGDP_y1_tst, PGDP_Mean_pred_1_tst, PGDP_nn_pred_1, PGDP_nn_t_pred_1  )
print_error(MSE_PGDP_nn_1, MAE_PGDP_nn_1, MSE_PGDP_Mean_pred_1, MAE_PGDP_Mean_pred_1, 'PGDP_1', MSE_PGDP_nn_t_1, MAE_PGDP_nn_t_1  )
plot_preds(PGDP_y2_tst, PGDP_Mean_pred_2_tst, PGDP_nn_pred_2, PGDP_nn_t_pred_2 )
print_error(MSE_PGDP_nn_2, MAE_PGDP_nn_2, MSE_PGDP_Mean_pred_2, MAE_PGDP_Mean_pred_2, 'PGDP_2', MSE_PGDP_nn_t_2, MAE_PGDP_nn_t_2  )
plot_preds(PGDP_y3_tst, PGDP_Mean_pred_3_tst, PGDP_nn_pred_3, PGDP_nn_t_pred_3 )
print_error(MSE_PGDP_nn_3, MAE_PGDP_nn_3, MSE_PGDP_Mean_pred_3, MAE_PGDP_Mean_pred_3, 'PGDP_3', MSE_PGDP_nn_t_3, MAE_PGDP_nn_t_3  )
plot_preds(PGDP_y4_tst, PGDP_Mean_pred_4_tst, PGDP_nn_pred_4, PGDP_nn_t_pred_4 )
print_error(MSE_PGDP_nn_4, MAE_PGDP_nn_4, MSE_PGDP_Mean_pred_4, MAE_PGDP_Mean_pred_4, 'PGDP_4', MSE_PGDP_nn_t_4, MAE_PGDP_nn_t_4  )

#%%

#results to excel

# .DataFrame is a constructor

# create a dictionary
results_dic = {
    'Indicator': ['MSE', 'MAE'],    
    'RGDP_1': [MSE_RGDP_nn_1, MAE_RGDP_nn_1],
    'RGDP_2': [MSE_RGDP_nn_2, MAE_RGDP_nn_2],
    'RGDP_3': [MSE_RGDP_nn_3, MAE_RGDP_nn_3],
    'RGDP_4': [MSE_RGDP_nn_4, MAE_RGDP_nn_4],
    'UNEMP_1': [MSE_UNEMP_nn_1, MAE_UNEMP_nn_1],
    'UNEMP_2': [MSE_UNEMP_nn_2, MAE_UNEMP_nn_2],
    'UNEMP_3': [MSE_UNEMP_nn_3, MAE_UNEMP_nn_3],
    'UNEMP_4': [MSE_UNEMP_nn_4, MAE_UNEMP_nn_4],
    'CPROF_1': [MSE_CPROF_nn_1, MAE_CPROF_nn_1],
    'CPROF_2': [MSE_CPROF_nn_2, MAE_CPROF_nn_2],
    'CPROF_3': [MSE_CPROF_nn_3, MAE_CPROF_nn_3],
    'CPROF_4': [MSE_CPROF_nn_4, MAE_CPROF_nn_4],
    'HOUSING_1': [MSE_HOUSING_nn_1, MAE_HOUSING_nn_1],
    'HOUSING_2': [MSE_HOUSING_nn_2, MAE_HOUSING_nn_2],
    'HOUSING_3': [MSE_HOUSING_nn_3, MAE_HOUSING_nn_3],
    'HOUSING_4': [MSE_HOUSING_nn_4, MAE_HOUSING_nn_4],
    'INDPROD_1': [MSE_INDPROD_nn_1, MAE_INDPROD_nn_1],
    'INDPROD_2': [MSE_INDPROD_nn_2, MAE_INDPROD_nn_2],
    'INDPROD_3': [MSE_INDPROD_nn_3, MAE_INDPROD_nn_3],
    'INDPROD_4': [MSE_INDPROD_nn_4, MAE_INDPROD_nn_4],
    'NGDP_1': [MSE_NGDP_nn_1, MAE_NGDP_nn_1],
    'NGDP_2': [MSE_NGDP_nn_2, MAE_NGDP_nn_2],
    'NGDP_3': [MSE_NGDP_nn_3, MAE_NGDP_nn_3],
    'NGDP_4': [MSE_NGDP_nn_4, MAE_NGDP_nn_4],
    'PGDP_1': [MSE_PGDP_nn_1, MAE_PGDP_nn_1],
    'PGDP_2': [MSE_PGDP_nn_2, MAE_PGDP_nn_2],
    'PGDP_3': [MSE_PGDP_nn_3, MAE_PGDP_nn_3],
    'PGDP_4': [MSE_PGDP_nn_4, MAE_PGDP_nn_4],   
    'RGDP_1_t': [MSE_RGDP_nn_t_1, MAE_RGDP_nn_t_1],
    'RGDP_2_t': [MSE_RGDP_nn_t_2, MAE_RGDP_nn_t_2],
    'RGDP_3_t': [MSE_RGDP_nn_t_3, MAE_RGDP_nn_t_3],
    'RGDP_4_t': [MSE_RGDP_nn_t_4, MAE_RGDP_nn_t_4],
    'UNEMP_1_t': [MSE_UNEMP_nn_t_1, MAE_UNEMP_nn_t_1],
    'UNEMP_2_t': [MSE_UNEMP_nn_t_2, MAE_UNEMP_nn_t_2],
    'UNEMP_3_t': [MSE_UNEMP_nn_t_3, MAE_UNEMP_nn_t_3],
    'UNEMP_4_t': [MSE_UNEMP_nn_t_4, MAE_UNEMP_nn_t_4],
    'CPROF_1_t': [MSE_CPROF_nn_t_1, MAE_CPROF_nn_t_1],
    'CPROF_2_t': [MSE_CPROF_nn_t_2, MAE_CPROF_nn_t_2],
    'CPROF_3_t': [MSE_CPROF_nn_t_3, MAE_CPROF_nn_t_3],
    'CPROF_4_t': [MSE_CPROF_nn_t_4, MAE_CPROF_nn_t_4],
    'HOUSING_1_t': [MSE_HOUSING_nn_t_1, MAE_HOUSING_nn_t_1],
    'HOUSING_2_t': [MSE_HOUSING_nn_t_2, MAE_HOUSING_nn_t_2],
    'HOUSING_3_t': [MSE_HOUSING_nn_t_3, MAE_HOUSING_nn_t_3],
    'HOUSING_4_t': [MSE_HOUSING_nn_t_4, MAE_HOUSING_nn_t_4],
    'INDPROD_1_t': [MSE_INDPROD_nn_t_1, MAE_INDPROD_nn_t_1],
    'INDPROD_2_t': [MSE_INDPROD_nn_t_2, MAE_INDPROD_nn_t_2],
    'INDPROD_3_t': [MSE_INDPROD_nn_t_3, MAE_INDPROD_nn_t_3],
    'INDPROD_4_t': [MSE_INDPROD_nn_t_4, MAE_INDPROD_nn_t_4],
    'NGDP_1_t': [MSE_NGDP_nn_t_1, MAE_NGDP_nn_t_1],
    'NGDP_2_t': [MSE_NGDP_nn_t_2, MAE_NGDP_nn_t_2],
    'NGDP_3_t': [MSE_NGDP_nn_t_3, MAE_NGDP_nn_t_3],
    'NGDP_4_t': [MSE_NGDP_nn_t_4, MAE_NGDP_nn_t_4],
    'PGDP_1_t': [MSE_PGDP_nn_t_1, MAE_PGDP_nn_t_1],
    'PGDP_2_t': [MSE_PGDP_nn_t_2, MAE_PGDP_nn_t_2],
    'PGDP_3_t': [MSE_PGDP_nn_t_3, MAE_PGDP_nn_t_3],
    'PGDP_4_t': [MSE_PGDP_nn_t_4, MAE_PGDP_nn_t_4],
    'RGDP_1_mean': [MSE_RGDP_Mean_pred_1, MAE_RGDP_Mean_pred_1],
    'RGDP_2_mean': [MSE_RGDP_Mean_pred_2, MAE_RGDP_Mean_pred_2],
    'RGDP_3_mean': [MSE_RGDP_Mean_pred_3, MAE_RGDP_Mean_pred_3],
    'RGDP_4_mean': [MSE_RGDP_Mean_pred_4, MAE_RGDP_Mean_pred_4],
    'UNEMP_1_mean': [MSE_UNEMP_Mean_pred_1, MAE_UNEMP_Mean_pred_1],
    'UNEMP_2_mean': [MSE_UNEMP_Mean_pred_2, MAE_UNEMP_Mean_pred_2],
    'UNEMP_3_mean': [MSE_UNEMP_Mean_pred_3, MAE_UNEMP_Mean_pred_3],
    'UNEMP_4_mean': [MSE_UNEMP_Mean_pred_4, MAE_UNEMP_Mean_pred_4],
    'CPROF_1_mean': [MSE_CPROF_Mean_pred_1, MAE_CPROF_Mean_pred_1],
    'CPROF_2_mean': [MSE_CPROF_Mean_pred_2, MAE_CPROF_Mean_pred_2],
    'CPROF_3_mean': [MSE_CPROF_Mean_pred_3, MAE_CPROF_Mean_pred_3],
    'CPROF_4_mean': [MSE_CPROF_Mean_pred_4, MAE_CPROF_Mean_pred_4],
    'HOUSING_1_mean': [MSE_HOUSING_Mean_pred_1, MAE_HOUSING_Mean_pred_1],
    'HOUSING_2_mean': [MSE_HOUSING_Mean_pred_2, MAE_HOUSING_Mean_pred_2],
    'HOUSING_3_mean': [MSE_HOUSING_Mean_pred_3, MAE_HOUSING_Mean_pred_3],
    'HOUSING_4_mean': [MSE_HOUSING_Mean_pred_4, MAE_HOUSING_Mean_pred_4],
    'INDPROD_1_mean': [MSE_INDPROD_Mean_pred_1, MAE_INDPROD_Mean_pred_1],
    'INDPROD_2_mean': [MSE_INDPROD_Mean_pred_2, MAE_INDPROD_Mean_pred_2],
    'INDPROD_3_mean': [MSE_INDPROD_Mean_pred_3, MAE_INDPROD_Mean_pred_3],
    'INDPROD_4_mean': [MSE_INDPROD_Mean_pred_4, MAE_INDPROD_Mean_pred_4],
    'NGDP_1_mean': [MSE_NGDP_Mean_pred_1, MAE_NGDP_Mean_pred_1],
    'NGDP_2_mean': [MSE_NGDP_Mean_pred_2, MAE_NGDP_Mean_pred_2],
    'NGDP_3_mean': [MSE_NGDP_Mean_pred_3, MAE_NGDP_Mean_pred_3],
    'NGDP_4_mean': [MSE_NGDP_Mean_pred_4, MAE_NGDP_Mean_pred_4],
    'PGDP_1_mean': [MSE_PGDP_Mean_pred_1, MAE_PGDP_Mean_pred_1],
    'PGDP_2_mean': [MSE_PGDP_Mean_pred_2, MAE_PGDP_Mean_pred_2],
    'PGDP_3_mean': [MSE_PGDP_Mean_pred_3, MAE_PGDP_Mean_pred_3],
    'PGDP_4_mean': [MSE_PGDP_Mean_pred_4, MAE_PGDP_Mean_pred_4], 
    
    
    
}

# create a list of strings
columns = ['Indicator', 'MSE', 'MAE']


# Passing a dictionary
# key: column name
# value: series of values
df_results = pd.DataFrame.from_dict(results_dic)

writer = pd.ExcelWriter('results_5.xlsx')
df_results.to_excel(writer,'results5')
writer.save()