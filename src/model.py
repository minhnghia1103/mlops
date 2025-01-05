import torchvision
import torch.nn as nn

def model():
    model = torchvision.models.vgg16(pretrained=True)
    # Replace the classifier for fine-tuning (optional)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)
    return model