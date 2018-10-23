def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
def compile_n_fit(validation_percent, testing_percent, image_height, image_width, n_channels, load_wt,dropout = 0.3, model_name = 'vgg16_model', magnification = '40X'):
    training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels = data_split(magnification = magnification, validation_percent = validation_percent, testing_percent = testing_percent)
    for i in range(len(models)):
        if models[i].__name__ == model_name:
            base_model = models[i]
    
    base_model = base_model(image_height=image_height,image_width=image_width,n_channels=n_channels,load_wt=load_wt)
    
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
    
    try:
        model.load_weights(model_name + '_weight_1.h5')
        print('Weights loaded!')
    except:
        print('No weights defined!')
        pass
    
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001), metrics=[f1,'accuracy'])
    early_stopping = EarlyStopping(patience=10, verbose=2)
    model_checkpoint = ModelCheckpoint(model_name + "_combine" +".model", save_best_only=True, verbose=2)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, verbose=2) #min_lr=0.00001,

    epochs = 100
    batch_size = 64

    history = model.fit(training_images, training_labels,
                        validation_data=[validation_images, validation_labels], 
                        epochs=epochs,
                        verbose = 0,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr])

    test_loss, test_acc, test_f1 = model.evaluate(testing_images, testing_labels)
    
    model.save_weights(model_name + '_weight_1.h5')
    
    print("\nThe test accuracy for " + model_name + " with magnification "+ magnification +" is ", test_acc, " with F1 score of ", test_f1, "\n")
