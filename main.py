import numpy as np 
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from models import *
from training import *
from keras.layers import *
from keras.models import *
from keras import layers
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

magnification_list = ['40X', '100X', '200X', '400X']
benign_list = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
malignant_list = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
cancer_list = benign_list + malignant_list

models = [vgg16_model, vgg19_model, xception_model, resnet_model, inception_model, inception_resnet_model]

'''
If you are getting memory problem while running the for loop below, remove the for loop which iterates for each models.
And input required the model name in the attribute of compile_n_fit function. eg: model_name = "inception_model"
'''

for model in models:
  iteration = 0
  for types in magnification_list:
      if iteration == 0:
          load_wt = "Yes"
      else:
          load_wt = "No"
      compile_n_fit(validation_percent=0.15, testing_percent=0.15,
                    image_height=115, image_width=175, n_channels=3, dropout = 0.3,
                    load_wt=load_wt, model_name = model.__name__, magnification = types)
      iteration += 1
