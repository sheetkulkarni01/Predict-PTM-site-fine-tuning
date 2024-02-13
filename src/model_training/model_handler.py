import torch

from model_components.T5Encoder import PT5_classification_model

def save_model(model,filepath):
# Saves all parameters that were changed during finetuning

    # Create a dictionary to hold the non-frozen parameters
    non_frozen_params = {}

    # Iterate through all the model parameters
    for param_name, param in model.named_parameters():
        # If the parameter has requires_grad=True, add it to the dictionary
        if param.requires_grad:
            non_frozen_params[param_name] = param

    # Save only the finetuned parameters 
    torch.save(non_frozen_params, filepath)

    
def load_model(filepath, num_labels=2):
# Creates a new PT5 model and loads the finetuned weights from a file

    # load a new model
    model, tokenizer = PT5_classification_model(num_labels=num_labels)
    
    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath)

    # Assign the non-frozen parameters to the corresponding parameters of the model
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data

    return tokenizer, model

def check_model_weights(model, model_reload):
    # Put both models to the same device
    model=model.to("cpu")
    model_reload=model_reload.to("cpu")

    # Iterate through the parameters of the two models and compare the data
    for param1, param2 in zip(model.parameters(), model_reload.parameters()):
        if not torch.equal(param1.data, param2.data):
            print("Models have different weights")
            break
    else:
        print("Models have identical weights")


def torch_save(model, filepath):
    torch.save(model.state_dict(), filepath)
