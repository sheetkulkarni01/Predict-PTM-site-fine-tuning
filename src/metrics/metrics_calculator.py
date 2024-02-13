import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, accuracy_score
from tabulate import tabulate

from model_training.train_model import create_dataset

def metrics_calculation(model, tokenizer, my_test):
    # Set the device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # create Dataset
    test_set=create_dataset(tokenizer,list(my_test['sequence']),list(my_test['label']))
    # make compatible with torch DataLoader
    test_set = test_set.with_format("torch", device=device)

    # Create a dataloader for the test dataset
    test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False)

    # Put the model in evaluation mode
    model.eval()

    # Make predictions on the test dataset
    raw_logits = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # add batch results (logits) to predictions
            raw_logits += model(input_ids, attention_mask=attention_mask).logits.tolist()
            labels += batch["labels"].tolist()

    # Convert logits to predictions
    raw_logits = np.array(raw_logits)
    predictions = np.argmax(raw_logits, axis=1)

    # Calculate metrics
    conf_matrix = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = conf_matrix.ravel()

    mcc = matthews_corrcoef(labels, predictions)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    accuracy = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, raw_logits[:, 1])  # Assuming binary classification, adjust accordingly


    metrics_table = [
        ["MCC", "Specificity", "Sensitivity", "Accuracy", "ROC-AUC"],
        [mcc, specificity, sensitivity, accuracy, roc_auc]
    ]

    print(tabulate(metrics_table, headers="firstrow", tablefmt="grid"))
    print(conf_matrix)