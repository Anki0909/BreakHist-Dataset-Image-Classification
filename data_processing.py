def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
    
def data_split(magnification = '40X', validation_percent = 0.15, testing_percent = 0.15):
    validation_percent = validation_percent
    testing_percent = testing_percent
    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []
    testing_images = []
    testing_labels = []
    for root, dirnames, filenames in os.walk("../input/breakhist_dataset/BreakHist_Dataset/" + magnification):
        if filenames == []:
            continue
        else:
            str_length = len("../input/breakhist_dataset/BreakHist_Dataset/40X/")
            #print(root)
            if root[str_length:str_length+6] == 'Benign':
                string_end = 56
            elif root[str_length:str_length+9] == 'Malignant':
                string_end = 59
            elif root[str_length+1:str_length+7] == 'Benign':
                string_end = 57
            else:
                string_end = 60
            name = root[string_end:]
            #print(name)
            #print(cancer_list.index(name))
            total_images = 0
            for names in filenames:
                total_images += 1
            print(name, magnification, total_images)
            validation_size = np.int(total_images*validation_percent)
            testing_size = np.int(total_images*testing_percent)
            training_size = total_images - (validation_size + testing_size)
            print(training_size, validation_size, testing_size, total_images)
            num = 0
            for names in filenames:
                num += 1
                filepath = os.path.join(root, names)
                #print(filepath)
                image = mpimg.imread(filepath)
                #if not all(image.shape == np.array([460,700,3])):
                #    print(names)
                #else:
                #    continue
                image_resize = resize(image,(115,175), mode = 'constant')
                if num in range(training_size):
                    training_images.append(image_resize[:,:,:])
                    training_labels.append(cancer_list.index(name))
                elif num in range(training_size,training_size+validation_size):
                    validation_images.append(image_resize[:,:,:])
                    validation_labels.append(cancer_list.index(name))
                elif num in range(training_size+validation_size,total_images):
                    testing_images.append(image_resize[:,:,:])
                    testing_labels.append(cancer_list.index(name))
    
    training_images = np.asarray(training_images)
    validation_images = np.asarray(validation_images)
    testing_images = np.asarray(testing_images)

    training_labels = np.asarray(training_labels)
    validation_labels = np.asarray(validation_labels)
    testing_labels = np.asarray(testing_labels)

    labels_count = np.unique(training_labels).shape[0]
    
    training_labels = dense_to_one_hot(training_labels, labels_count)
    training_labels = training_labels.astype(np.float32)
    validation_labels = dense_to_one_hot(validation_labels, labels_count)
    validation_labels = validation_labels.astype(np.float32)
    testing_labels = dense_to_one_hot(testing_labels, labels_count)
    testing_labels = testing_labels.astype(np.float32)
    print(training_images.shape[0],validation_images.shape[0],testing_images.shape[0])
    
    return training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels
