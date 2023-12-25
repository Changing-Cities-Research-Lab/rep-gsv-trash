import torch
import torch.nn as nn
from torchvision import datasets, models, transforms



class Classifier(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    """
    Args:
    
    num_classes: Output number of classes for classification
    
    Initiates model with resnet50 backbone and own classifier layers with output of num_classes
    """
    self.backbone = models.resnet50(pretrained=True)
    num_features = self.backbone.fc.out_features
    self.classifier_layer = nn.Sequential(
      nn.Linear(num_features, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
      nn.Linear(256, 128),
      nn.ReLU(inplace=True),
    )
    #Classifier layer adds complexity to the final output layer which performed better in trials.

    #Svm is simple linear layer at end. This is no longer representative of SVM with image embedding extraction
    #but trained model requires the name to still be named svm.
    self.svm = nn.Linear(128, num_classes)

  def forward(self, x):
    h = self.backbone(x)
    h = self.classifier_layer(h)
    h = self.svm(h)
    return h
