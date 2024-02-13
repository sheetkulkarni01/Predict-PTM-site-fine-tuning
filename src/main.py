import os.path

import torch
import numpy as np
import transformers, datasets

from Bio import SeqIO
import pandas as pd

from sklearn.model_selection import train_test_split

from visualization.plotting import plot_training
from model_training.train_model import train_per_protein
from visualization.training_history_plot import get_train_per_protein_history
from model_training.model_handler import torch_save, load_model, check_model_weights
from metrics.metrics_calculator import metrics_calculation
from neural_network_architectures.cnn_with_embeddings import load_embeddings, train_cnn_embeddings
from neural_network_architectures.cnn_with_sequences import get_encoding, train_cnn_sequences
from neural_network_architectures.transformer import train_transformer 

def fasta_sequence(file_path)->list:
    # Load FASTA file using Biopython
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        # Split the description to extract label
        description_parts = record.description.split("%")
        label = int(description_parts[-1].split("LABEL=")[1])  # Extracting the numeric part of the label
        sequences.append([record.name, str(record.seq), label])
    return sequences

if __name__ == "__main__":
    os.chdir("/scratch/users/sheetal/ProtTrans/Fine-Tuning")
    # Set environment variables to run Deepspeed from a notebook
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    print("Torch version: ",torch.__version__)
    print("Cuda version: ",torch.version.cuda)
    print("Numpy version: ",np.__version__)
    print("Pandas version: ",pd.__version__)
    print("Transformers version: ",transformers.__version__)
    print("Datasets version: ",datasets.__version__)

    # Create Train dataframe
    df = pd.DataFrame(fasta_sequence('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/input_datasets/train_Pos_Neg_Y.fasta'), columns=["name", "sequence", "label"])
    # Display the dataframe
    df.head(5)
    # Split the dataset into training and validation sets
    my_train, my_valid = train_test_split(df, test_size=0.2, random_state=42)

    my_train=my_train[["sequence", "label"]]
    my_valid=my_valid[["sequence","label"]]

    # Print the first 5 rows of the training & validation set
    print("Training Set: ", my_train.head(5))
    print("\nValidation Set:", my_valid.head(5))

    # Train the model
    tokenizer, model, history = train_per_protein(my_train, my_valid, num_labels=2, batch=1, accum=8, epochs=20, seed=42)
    get_train_per_protein_history(history)

    torch_save(model, "./my_finetuned.pth")

    tokenizer, model_reload = load_model("./my_finetuned.pth", num_labels=2)

    check_model_weights(model, model_reload)

    # Create Test dataframe
    df = pd.DataFrame(fasta_sequence('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/input_datasets/test_Pos_Neg_Y.fasta'), columns=["name", "sequence", "label"])
    # Display the dataframe
    df.head(5)

    my_test=df[["sequence", "label"]]

    #Using .loc ensures that you are modifying the original DataFrame rather than a view of it, which helps avoid the SettingWithCopyWarning.
    # Replace characters in the "sequence" column
    my_test.loc[:, "sequence"] = my_test["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
    # Convert each sequence to a space-separated string
    my_test.loc[:, 'sequence'] = my_test.apply(lambda row: " ".join(row["sequence"]), axis=1)

    metrics_calculation(model, tokenizer, my_test)

    my_train.to_csv('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/csv_dataset_files/my_train.csv', index=False)
    my_valid.to_csv('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/csv_dataset_files/my_valid.csv', index=False)
    my_test.to_csv('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/csv_dataset_files/my_test.csv', index=False)

    train_embeddings = load_embeddings('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/extracted_embeddings/h5_files/data_train.h5')    
    valid_embeddings = load_embeddings('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/extracted_embeddings/h5_files/data_valid.h5')
    test_embeddings = load_embeddings('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/extracted_embeddings/h5_files/data_test.h5')

    pre_train_Y_CNN_labels = np.array(my_train['label'])
    pre_valid_Y_CNN_labels = np.array(my_valid['label'])
    pre_test_Y_CNN_labels = np.array(my_test['label'])

    # train and plot CNN with embeddings
    pre_his, input_seq_l, all_train_acc, all_val_acc = train_cnn_embeddings(train_embeddings, valid_embeddings, pre_train_Y_CNN_labels, pre_valid_Y_CNN_labels)
    plot_training("cnn_with_embeddings", pre_his, input_seq_l, all_train_acc, all_val_acc, test_embeddings, pre_test_Y_CNN_labels)

    pre_train_Y_CNN = get_encoding('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/csv_dataset_files/my_train.csv', 'sequence')
    pre_valid_Y_CNN = get_encoding('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/csv_dataset_files/my_valid.csv', 'sequence')
    pre_test_Y_CNN = get_encoding('/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/src/csv_dataset_files/my_test.csv', 'sequence')

    # train and plot CNN with sequences
    pre_his, all_train_acc, all_val_acc = train_cnn_sequences(pre_train_Y_CNN, pre_valid_Y_CNN, pre_train_Y_CNN_labels, pre_valid_Y_CNN_labels)
    plot_training("cnn_with_sequences", pre_his, 0, all_train_acc, all_val_acc, pre_test_Y_CNN, pre_test_Y_CNN_labels)

    # train and plot Transformer
    pre_his, all_train_acc, all_val_acc = train_transformer(pre_train_Y_CNN, pre_valid_Y_CNN, pre_train_Y_CNN_labels, pre_valid_Y_CNN_labels)
    plot_training("transformer", pre_his, 0, all_train_acc, all_val_acc, pre_test_Y_CNN, pre_test_Y_CNN_labels)
