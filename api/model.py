import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural Network


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 768)
        self.out = nn.Linear(768, num_classes)

    def forward(self, features, labels=None):
        x = F.relu(self.fc1(features.to(device)))
        x = F.relu(self.fc2(x))
        logits = self.out(x)

        # Calculate loss (for validation)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        # Trainer needs this for training
        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_model():
    path_root = os.getenv('PATH_ROOT')
    model_path = os.path.join(path_root, 'api', 'temp.pth')
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = NN(input_size=10, num_classes=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def predict(model, input_lexical):
    # Disable gradient computation for inference
    with torch.no_grad():
        predictions = model(input_lexical)
    # Get the predicted class (if classification task)
    predicted_class = torch.argmax(predictions.logits, dim=1)
    return predicted_class
