import numpy as np

import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, accuracy_score

from neural_network_architectures.cnn_with_embeddings import cnn_model_embeddings
from neural_network_architectures.cnn_with_sequences import cnn_model_sequences
from neural_network_architectures.transformer import transformer_model

def plot(history)->None:
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_training(model, pre_history, input_sequence_length, all_train_accuracies, all_val_accuracies, pre_test, pre_test_y_cnn_labels)->None:
    # Calculate average metrics
    # avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_train_accuracy = np.mean(all_train_accuracies, axis=0)
    # avg_val_loss = np.mean(all_val_losses, axis=0)
    avg_val_accuracy = np.mean(all_val_accuracies, axis=0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 2)
    plt.plot(avg_train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(avg_val_accuracy, label='Validation Accuracy', color='red')
    plt.title('Average Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    if (model == "cnn_with_embeddings"):
        loaded_model = cnn_model_embeddings(input_sequence_length)
        loaded_model.load_weights("./pre_model_cnn_embedding.h5")
        y_pred = loaded_model.predict(pre_test).reshape(pre_test_y_cnn_labels.shape[0],)
        plt.savefig(f"/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/Plots/cnn_with_embeddings.pdf")
    elif (model == "cnn_with_sequences"):
        loaded_model = cnn_model_sequences()
        loaded_model.load_weights("./pre_model_cnn_seq.h5")
        y_pred = loaded_model.predict(pre_test).reshape(pre_test_y_cnn_labels.shape[0],)
        plt.savefig(f"/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/Plots/cnn_with_sequences.pdf")
    elif (model == "transformer"):
        loaded_model = transformer_model()
        loaded_model.load_weights("./pre_model_transformer.h5")
        y_pred = loaded_model.predict(pre_test).reshape(pre_test_y_cnn_labels.shape[0],)
        plt.savefig(f"/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/Plots/transformer_with_sequences.pdf")
    else:
        print("Undefined model")

    y_pred = (y_pred > 0.5)
    y_pred = [int(i) for i in y_pred]
    pre_test_y_cnn_labels = np.array(pre_test_y_cnn_labels)
    y_pred = np.array(y_pred)

    # Evaluate CNN model on test set
    cm = confusion_matrix(pre_test_y_cnn_labels, y_pred)
    mcc = matthews_corrcoef(pre_test_y_cnn_labels, y_pred)
    acc = accuracy_score(pre_test_y_cnn_labels, y_pred)
    roc_auc = roc_auc_score(pre_test_y_cnn_labels, y_pred)

    sn_cnn = cm[1][1] / (cm[1][1] + cm[1][0])
    sp_cnn = cm[0][0] / (cm[0][0] + cm[0][1])

    plot(pre_history)

    table_data = [
        ["Metric", "Value"],
        ["Accuracy", str(acc)],
        ["AUC", str(roc_auc)],
        ["Sensitivity", str(sn_cnn)],
        ["Specificity", str(sp_cnn)],
        ["MCC", str(mcc)],
    ]

    # Print the table
    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
