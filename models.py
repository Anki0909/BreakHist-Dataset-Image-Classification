def vgg16_model(image_height, image_width, n_channels, load_wt = "Yes"):
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    
    model = Sequential()
    #model.add(Input(shape=(115,175,3)))
    
    model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(image_height, image_width, n_channels)))
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    
    if load_wt == "Yes":
        model.load_weights(weights_path)
    
    return model
    
def vgg19_model(image_height, image_width, n_channels, load_wt = "Yes"):
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    
    img_input = Input(shape=(image_height, image_width, n_channels))
    
    x = Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv1')(img_input)
    x = Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv2')(x)
    x = MaxPooling2D((2,2),strides=(2,2),name='block1_pool')(x)
    
    x = Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv1')(x)
    x = Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv2')(x)
    x = MaxPooling2D((2,2),strides=(2,2),name='block2_pool')(x)
    
    x = Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv1')(x)
    x = Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv2')(x)
    x = Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv3')(x)
    x = Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv4')(x)
    x = MaxPooling2D((2,2),strides=(2,2),name='block3_pool')(x)
    
    x = Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv1')(x)
    x = Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv2')(x)
    x = Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv3')(x)
    x = Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv4')(x)
    x = MaxPooling2D((2,2),strides=(2,2),name='block4_pool')(x)
    
    x = Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv1')(x)
    x = Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv2')(x)
    x = Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv3')(x)
    x = Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv4')(x)
    x = MaxPooling2D((2,2),strides=(2,2),name='block5_pool')(x)
    x = Flatten()(x)
    
    inp = img_input
    
    model = Model(inp, x, name='vgg19')
    
    if load_wt == "Yes":
        model.load_weights(weights_path)
    
    return model
    
  def xception_model(image_height, image_width, n_channels, load_wt = "Yes"):
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    
    input_layer = Input((image_height, image_width, n_channels))
    x = Conv2D(32,(3,3),strides=(2,2),use_bias=False,name='block1_conv1')(input_layer)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu',name='block1_conv1_act')(x)
    x = Conv2D(64,(3,3),use_bias=False,name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu',name='block1_conv2_act')(x)
    
    residual = Conv2D(128,(1,1),strides=(2,2),padding='same',use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    x = SeparableConv2D(128,(3,3),padding='same',use_bias=False,name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu',name='block_sepconv1_act')(x)
    x = SeparableConv2D(128,(3,3),padding='same',use_bias=False,name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)
    
    x = MaxPooling2D((3,3),strides=(2,2),padding='same',name='block2_pool')(x)
    x = layers.add([x, residual])
    
    residual = Conv2D(256,(1,1),strides=(2,2),padding='same',use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    x = Activation('relu',name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256,(3,3),padding='same',use_bias=False,name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu',name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256,(3,3),padding='same',use_bias=False,name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)
    
    x = MaxPooling2D((3,3),strides=(2,2),padding='same',name='block3_pool')(x)
    x = layers.add([x,residual])
    
    residual = Conv2D(728,(1,1),strides=(2,2),use_bias=False,padding='same')(x)
    residual = BatchNormalization()(residual)
    
    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728,(3,3),padding='same',use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu',name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728,(3,3),padding='same',use_bias=False,name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)
    
    x = MaxPooling2D((3,3),strides=(2,2),padding='same',name='block4_pool')(x)
    x = layers.add([x,residual])
    
    for i in range(8):
        residual = x
        prefix = 'block' + str(i+5)
        
        x = Activation('relu', name=prefix+'_sepconv1_act')(x)
        x = SeparableConv2D(728,(3,3),padding='same',use_bias=False,name=prefix+'_sepconv1')(x)
        x = BatchNormalization(name=prefix+'_sepconv1_bn')(x)
        x = Activation('relu',name=prefix+'_sepconv2_act')(x)
        x = SeparableConv2D(728,(3,3),padding='same',use_bias=False,name=prefix+'_sepconv2')(x)
        x = BatchNormalization(name=prefix+'_sepconv2_bn')(x)
        x = Activation('relu',name=prefix+'_sepconv3_act')(x)
        x = SeparableConv2D(728,(3,3),padding='same',use_bias=False,name=prefix+'_sepconv3')(x)
        x = BatchNormalization(name=prefix+'_sepconv3_bn')(x)
        
        x = layers.add([x,residual])
        
    residual = Conv2D(1024,(1,1),strides=(2,2),padding='same',use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728,(3,3),padding='same',use_bias=False,name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu',name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024,(3,3),padding='same',use_bias=False,name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)
    
    x = MaxPooling2D((3,3),strides=(2,2),padding='same',name='block13_pool')(x)
    x = layers.add([x,residual])
    
    x = SeparableConv2D(1536,(3,3),padding='same',use_bias=False,name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu',name='block14_sepconv1_act')(x)
    
    x = SeparableConv2D(2048,(3,3),padding='same',use_bias=False,name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu',name='block14_sepconv2_act')(x)
    
    x = GlobalAveragePooling2D()(x)
    
    model = Model(input_layer,x,name='xception')
    
    if load_wt == "Yes":
        model.load_weights(weights_path)
    
    return model

def resnet_model(image_height, image_width, n_channels, load_wt = "Yes"):
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af')
    
    def identity_block(input_tensor,kernel_size,filters,stage,block):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res'+str(stage)+block+'_branch'
        bn_name_base = 'bn'+str(stage)+block+'_branch'
        
        x = Conv2D(filters1,(1,1),name=conv_name_base+'2a')(input_tensor)
        x = BatchNormalization(axis=3,name=bn_name_base+'2a')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
        x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
        x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)
        
        x = layers.add([x,input_tensor])
        x = Activation('relu')(x)
        
        return x
    
    def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):
        filters1,filters2,filters3 = filters
        conv_name_base = 'res'+str(stage)+block+'_branch'
        bn_name_base = 'bn'+str(stage)+block+'_branch'
    
        x = Conv2D(filters1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
        x = BatchNormalization(axis=3,name=bn_name_base+'2a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
        x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
        x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)
    
        shortcut = Conv2D(filters3, (1, 1), strides=strides,name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    img_input = Input(shape=(image_height, image_width, n_channels))
    img_padding = ZeroPadding2D(((41,41),(11,11)))(img_input)
    x = ZeroPadding2D((3,3))(img_padding)
    x = Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)
    x = BatchNormalization(axis=3,name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides=(2,2))(x)
    
    x = conv_block(x, 3, [64,64,256],stage=2,block='a',strides=(1,1))
    x = identity_block(x,3,[64,64,256],stage=2,block='b')
    x = identity_block(x,3,[64,64,256],stage=2,block='c')
    
    x = conv_block(x,3,[128,128,512],stage=3,block='a')
    x = identity_block(x,3,[128,128,512],stage=3,block='b')
    x = identity_block(x,3,[128,128,512],stage=3,block='c')
    x = identity_block(x,3,[128,128,512],stage=3,block='d')
    
    x = conv_block(x,3,[256,256,1024],stage=4,block='a')
    x = identity_block(x,3,[256,256,1024],stage=4,block='b')
    x = identity_block(x,3,[256,256,1024],stage=4,block='c')
    x = identity_block(x,3,[256,256,1024],stage=4,block='d')
    x = identity_block(x,3,[256,256,1024],stage=4,block='e')
    x = identity_block(x,3,[256,256,1024],stage=4,block='f')
    
    x = conv_block(x,3,[512,512,2048],stage=5,block='a')
    x = identity_block(x,3,[512,512,2048],stage=5,block='b')
    x = identity_block(x,3,[512,512,2048],stage=5,block='c')
    
    x = AveragePooling2D((7,7),name='avg_pool')(x)
    x = Flatten()(x)
    
    inp = img_input
    
    model = Model(inp,x,name='resnet50')
    
    if load_wt == "Yes":
        model.load_weights(weights_path)
    
    return model
    
def inception_model(image_height, image_width, n_channels, load_wt = "Yes"):
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
    
    def conv2d_bn(x,filters,num_row,num_col,padding='same',strides=(1,1),name=None):
        if name is not None:
            bn_name = name+'_bn'
            conv_name = name+'_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters,(num_row,num_col),strides=strides,padding=padding,use_bias=False,name=conv_name)(x)
        x = BatchNormalization(axis=3,scale=False,name=bn_name)(x)
        x = Activation('relu',name=name)(x)
        
        return x
    
    channel_axis = 3
    img_input = Input(shape=(image_height,image_width,n_channels))
    zero_pad = ZeroPadding2D(((12,12),(0,0)))(img_input)
    
    x = conv2d_bn(zero_pad, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
        
    x = GlobalMaxPooling2D()(x)
    
    inp = img_input
    
    model = Model(inp,x,name='inception_v3')
    
    if load_wt == "Yes":
        model.load_weights(weights_path)
    
    return model
    
def inception_resnet_model(image_height, image_width, n_channels, load_wt = "Yes"):
    BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'
    weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file(weights_filename, BASE_WEIGHT_URL + weights_filename, cache_subdir='models',md5_hash='d19885ff4a710c122648d3b5c3b684e4')
    
    def conv2d_bn(x,filters,kernel_size,strides=1,padding='same',activation='relu',use_bias=False,name=None):
        x = Conv2D(filters,kernel_size,strides=strides,padding=padding,use_bias=use_bias,name=name)(x)
        if not use_bias:
            bn_axis = 3
            bn_name = None if name is None else name + '_bn'
            x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        if activation is not None:
            ac_name = None if name is None else name + '_ac'
            x = Activation(activation, name=ac_name)(x)
        return x
    
    def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):

        if block_type == 'block35':
            branch_0 = conv2d_bn(x, 32, 1)
            branch_1 = conv2d_bn(x, 32, 1)
            branch_1 = conv2d_bn(branch_1, 32, 3)
            branch_2 = conv2d_bn(x, 32, 1)
            branch_2 = conv2d_bn(branch_2, 48, 3)
            branch_2 = conv2d_bn(branch_2, 64, 3)
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = conv2d_bn(x, 192, 1)
            branch_1 = conv2d_bn(x, 128, 1)
            branch_1 = conv2d_bn(branch_1, 160, [1, 7])
            branch_1 = conv2d_bn(branch_1, 192, [7, 1])
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = conv2d_bn(x, 192, 1)
            branch_1 = conv2d_bn(x, 192, 1)
            branch_1 = conv2d_bn(branch_1, 224, [1, 3])
            branch_1 = conv2d_bn(branch_1, 256, [3, 1])
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))

        block_name = block_type + '_' + str(block_idx)
        channel_axis = 3
        mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
        up = conv2d_bn(mixed,K.int_shape(x)[channel_axis],1,activation=None,use_bias=True,name=block_name + '_conv')
        
        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                   output_shape=K.int_shape(x)[1:],
                   arguments={'scale': scale},
                   name=block_name)([x, up])
        if activation is not None:
            x = Activation(activation, name=block_name + '_ac')(x)
        return x
    
    channel_axis = 3
    img_input = Input(shape=(image_height,image_width,n_channels))
    zero_pad = ZeroPadding2D(((12,12),(0,0)))(img_input)
    

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(zero_pad, 32, 3, strides=2, padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    x = GlobalMaxPooling2D()(x)

    inputs = img_input

    # Create model
    model = Model(inputs, x, name='inception_resnet_v2')

    if load_wt == "Yes":
        model.load_weights(weights_path)

    return model
