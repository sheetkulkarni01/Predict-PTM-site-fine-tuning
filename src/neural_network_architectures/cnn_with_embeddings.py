from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import h5py
import numpy as np


def load_embeddings(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        embeddings = hf['embedding'][:]
    return embeddings
 

def cnn_model_embeddings(input_sequence_length):
    model = Sequential()
    model.add(Conv1D(128, kernel_size=16, activation='relu', kernel_initializer='he_normal', input_shape=(input_sequence_length, 1)))
    model.add(Dropout(0.01))       
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.01))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_cnn_embeddings(train_embeddings, valid_embeddings, pre_train_y_cnn_labels, pre_valid_y_cnn_labels):
    num_training_loops = 5
    all_train_losses = []
    all_train_accuracies = []
    all_val_losses = []
    all_val_accuracies = []
    max_len = 0
    saved_results_list = []

    n_epo_cnn = 20
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc_roc')]

    train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], train_embeddings.shape[1])
    print(train_embeddings.shape)

    # Assuming valid_embeddings has a shape of (40, 1024)
    valid_embeddings = valid_embeddings.reshape(valid_embeddings.shape[0], valid_embeddings.shape[1])
    print(valid_embeddings.shape)

    # Create the CNN model with the correct input shape
    input_sequence_length = train_embeddings.shape[1]

    for i in range(num_training_loops):
        print(f"\nTraining Loop {i + 1}")

        model_cnn = cnn_model_embeddings(input_sequence_length)
        model_cnn.compile(optimizer=Adam(learning_rate=1e-5), loss=BinaryCrossentropy(), metrics=metrics)

        checkpointer = ModelCheckpoint(filepath="./pre_model_cnn_embedding.h5", 
                                monitor="val_accuracy",
                                verbose=0,
                                save_weights_only=True,
                                save_best_only=True)
        
        # Train the model using loaded embeddings
        pre_history_cnn_embed = model_cnn.fit(train_embeddings, pre_train_y_cnn_labels, batch_size=8, epochs=n_epo_cnn, verbose=1, callbacks=[checkpointer],
                                validation_data=(valid_embeddings, pre_valid_y_cnn_labels))

        all_train_losses.append(pre_history_cnn_embed.history['loss'])
        all_train_accuracies.append(pre_history_cnn_embed.history['accuracy'])
        all_val_losses.append(pre_history_cnn_embed.history['val_loss'])
        all_val_accuracies.append(pre_history_cnn_embed.history['val_accuracy'])

        # Save the results after each loop
        saved_results = {
            'train_losses': all_train_losses.copy(),
            'train_accuracies': all_train_accuracies.copy(),
            'val_losses': all_val_losses.copy(),
            'val_accuracies': all_val_accuracies.copy()
        }
        saved_results_list.append(saved_results)
        np.save(f'saved_results_loop_{i + 1}.py', saved_results)

        max_len = max(max_len, len(all_train_losses[-1])) #since arrays within the list are not of same shape, so making the length of the lists as same


    for result in saved_results_list:
        for key in result:
            result[key] = [np.pad(lst, (0, max_len - len(lst))) for lst in result[key]]


    # Trim or zero-pad the lists to the maximum length
    #all_train_losses = [lst[:max_len] for lst in all_train_losses]
    all_train_accuracies = [lst[:max_len] for lst in all_train_accuracies]
    #all_val_losses = [lst[:max_len] for lst in all_val_losses]
    all_val_accuracies = [lst[:max_len] for lst in all_val_accuracies]


    return pre_history_cnn_embed, input_sequence_length, all_train_accuracies, all_val_accuracies


    
