import numpy as np 
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from models import *
from training_fn import *
from keras.layers import *
from keras.models import *
from keras import layers
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

magnification_list = ['40X', '100X', '200X', '400X']
benign_list = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
malignant_list = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
cancer_list = benign_list + malignant_list

models = [vgg16_model, vgg19_model, xception_model, resnet_model, inception_model, inception_resnet_model]

model_num = 3
name = models[model_num].__name__

iteration = 0
for types in magnification_list:
  if iteration == 0:
    load_wt = "Yes"
  else:
    load_wt = "No"
  compile_n_fit(validation_percent=0.15, testing_percent=0.15,
                    image_height=115, image_width=175, n_channels=3, dropout = 0.3,
                    load_wt=load_wt, model_name = name, magnification = types)
  iteration += 1

dropout = 0.3
base_model = models[model_num]
base_model = base_model(image_height=115,image_width=175,n_channels=3,load_wt='No')
x = base_model.output
x = Dense(2048, activation = 'relu')(x)
x = Dropout(dropout)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(dropout)(x)
x = Dense(128, activation = 'relu')(x)
x = Dropout(dropout)(x)
x = Dense(32, activation = 'relu')(x)
out = Dense(8, activation = 'softmax')(x)
inp = base_model.input

model = Model(inp,out)

model.load_weights(name + '_weight_1.h5')

layer_name = None
for idx, layer in enumerate(model.layers):
    if layer.name[:7] == 'flatten' or layer.name[:6] == 'global':
        layer_name = layer.name
        break

model_fe = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

for types in magnification_list:
    training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels = data_split(magnification = types, validation_percent = 0.15, testing_percent = 0.15, encoding="No")

    training_features = model_fe.predict(training_images)
    validation_features = model_fe.predict(validation_images)
    testing_features = model_fe.predict(testing_images)
    
    lr = LogisticRegression()
    svm_l = SVC(kernel='linear')
    
    fs_model = SelectFromModel(ExtraTreesClassifier(n_estimators=50), prefit=False)
    training_features_new = fs_model.fit_transform(training_features, training_labels)
    validation_features_new = fs_model.transform(validation_features)
    testing_features_new = fs_model.transform(testing_features)
    
    classifier_list = [lr, svm_l]
    classifier_label = ['Logistic Regression', 'Linear SVM']
    
    scoring = ['accuracy', 'f1_weighted']
    print('Cross validation:')
    for classifier, label in zip(classifier_list, classifier_label):
        scores = cross_validate(estimator=classifier, X=training_features, y=training_labels, cv=10, scoring=scoring)
        print("[%s]\nAccuracy: %0.3f\tF1 Weighted: %0.3f"
                % (label, scores['test_accuracy'].mean(), scores['test_f1_weighted'].mean()))    
