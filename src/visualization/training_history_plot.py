import matplotlib.pyplot as plt

# Get loss, val_loss, and the computed metric from history
def get_train_per_protein_history(history):

    loss = [x['loss'] for x in history if 'loss' in x]
    val_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

    # Get spearman (for regression) or accuracy value (for classification)
    if [x['eval_spearmanr'] for x in history if 'eval_spearmanr' in x] != []:
        metric = [x['eval_spearmanr'] for x in history if 'eval_spearmanr' in x]
    else:
        metric = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]

    epochs = [x['epoch'] for x in history if 'loss' in x]

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Plot loss and val_loss on the first y-axis
    line1 = ax1.plot(epochs, loss, label='train_loss')
    line2 = ax1.plot(epochs, val_loss, label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Plot the computed metric on the second y-axis
    line3 = ax2.plot(epochs, metric, color='red', label='val_metric')
    ax2.set_ylabel('Metric')
    ax2.set_ylim([0, 1])

    # Combine the lines from both y-axes and create a single legend
    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower left')

    # Show the plot
    plt.title("Training History")
    plt.savefig(f"/scratch/users/sheetal/ProtTrans/Fine-Tuning/Predict-PTM-site-fine-tuning/Plots/training_history.pdf")
    plt.show()
