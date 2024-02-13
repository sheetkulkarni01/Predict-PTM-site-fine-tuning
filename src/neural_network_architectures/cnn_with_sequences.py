from keras.models import Sequential
from keras.layers import Embedding, Lambda, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd

#CNN with sequences  
def get_encoding(csv_file, sequence):
    # This is returning just integers to feed to the embedding layer
    
    encodings = []
    
    # define universe of possible input values
    alphabet = 'ARNDCQEGHILKMFPSTWYV-'
    
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    
    data = pd.read_csv(csv_file)
    sequences = data[sequence].tolist()
    
    for seq in sequences:
        try:
            integer_encoded = [char_to_int[char] for char in seq if char in alphabet]
        except:
            continue
        encodings.append(integer_encoded)
        
    encodings = np.array(encodings)
    print(encodings.shape)
    return encodings

def cnn_model_sequences():
    model = Sequential()
    model.add(Embedding(256, 21, input_length=33))
    model.add(Lambda(lambda x: tf.expand_dims(x, 3)))
    model.add(Conv2D(128, kernel_size=(16, 3), activation = 'relu', kernel_initializer='he_normal', padding = 'VALID'))
    model.add(Dropout(0.02))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(8, 3), activation = 'relu', kernel_initializer='he_normal', padding = 'VALID'))
    model.add(Dropout(0.02))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.02))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_cnn_sequences(pre_train_y_cnn, pre_valid_y_cnn, pre_train_y_cnn_labels, pre_valid_y_cnn_labels):
    n_epo_cnn = 20
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc_roc')]


    all_train_losses_cnn = []
    all_train_accuracies_cnn = []
    all_val_losses_cnn = []
    all_val_accuracies_cnn = []

    max_len_cnn = 0
    saved_results_list_cnn = []

    for i in range(5):
        print(f"\nTraining Loop {i + 1}")

        model_seq = cnn_model_sequences()
        model_seq.compile(optimizer=Adam(learning_rate=1e-5), loss=BinaryCrossentropy(), metrics=metrics)

        checkpointer = ModelCheckpoint(filepath="./pre_model_cnn_seq.h5", 
                                    monitor="val_accuracy",
                                    verbose=0,
                                    save_weights_only=True,
                                    save_best_only=True)

        
        # Train the model using loaded embeddings
        pre_history_seq = model_seq.fit(pre_train_y_cnn, pre_train_y_cnn_labels, batch_size=32, epochs=n_epo_cnn, verbose=1, callbacks=[checkpointer],
                                validation_data=(pre_valid_y_cnn, pre_valid_y_cnn_labels))

        

        all_train_losses_cnn.append(pre_history_seq.history['loss'])
        all_train_accuracies_cnn.append(pre_history_seq.history['accuracy'])
        all_val_losses_cnn.append(pre_history_seq.history['val_loss'])
        all_val_accuracies_cnn.append(pre_history_seq.history['val_accuracy'])

        # Save the results after each loop
        saved_results = {
            'train_losses': all_train_losses_cnn.copy(),
            'train_accuracies': all_train_accuracies_cnn.copy(),
            'val_losses': all_val_losses_cnn.copy(),
            'val_accuracies': all_val_accuracies_cnn.copy()
        }
        saved_results_list_cnn.append(saved_results)
        np.save(f'saved_results_loop_{i + 1}.npy', saved_results)

        max_len_cnn = max(max_len_cnn, len(all_train_losses_cnn[-1])) #since arrays within the list are not of same shape, so making the length of the lists as same


    for result in saved_results_list_cnn:
        for key in result:
            result[key] = [np.pad(lst, (0, max_len_cnn - len(lst))) for lst in result[key]]


    # Trim or zero-pad the lists to the maximum length
    all_train_losses_cnn = [lst[:max_len_cnn] for lst in all_train_losses_cnn]
    all_train_accuracies_cnn_seq = [lst[:max_len_cnn] for lst in all_train_accuracies_cnn]
    all_val_losses_cnn = [lst[:max_len_cnn] for lst in all_val_losses_cnn]
    all_val_accuracies_cnn_seq = [lst[:max_len_cnn] for lst in all_val_accuracies_cnn]

    return pre_history_seq, all_train_accuracies_cnn_seq, all_val_accuracies_cnn_seq
