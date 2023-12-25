# https://github.com/christiansafka/img2vec
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from model import Classifier
import numpy as np
from tqdm import tqdm
import os
import csv

class Img2Vec:

    def __init__(self, cuda=False, model='resnet-50', layer='default', layer_output_size=2048, num_output_labels=2):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer


        Img2vec is a class that takes a model and utilizes that model's architecture

        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer, num_output_labels)

        self.model = self.model.to(self.device)

        self.model.eval()

    import csv

    import csv
    import numpy as np
    import csv
    import numpy as np

    def get_vec(self, dataloader, tensor=False, output_name=None, add_name = False):
        """
         Get vector embeddings from model's output before pooling layers
         Takes in images from a dataloader (allows for faster multiprocessing with GPU) and saves to CSV

        :param dataloader: PyTorch DataLoader
        :param tensor: If True, get_vec will return a FloatTensor instead of a Numpy array
        :param output_name: Name of the CSV file to save the embeddings and labels
        :param add_name: If True, get_vec will also track the name of the images.
        :returns: Numpy ndarray or FloatTensor of extracted embeddings
        """
        embeddings = []
        labels = []
        image_names = []
        predicts = []

        for images, label, paths in tqdm(dataloader):
            print(paths)
            images = images.to(self.device)

            my_embedding = torch.zeros(images.size(0), self.layer_output_size, 1, 1).to(self.device)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(images)
            _ , predict = torch.max(h_x, dim =1)
            h.remove()


            predicts.extend(predict)
            if tensor:
                embeddings.append(my_embedding)
            else:
                embeddings.extend(my_embedding.cpu().numpy()[:, :, 0, 0].tolist())  # Extend instead of append
            labels.extend(label.numpy().tolist())  # Extend instead of append
            if add_name:
                image_names.extend([os.path.basename(path) for path in paths])

        embeddings = np.array(embeddings) if tensor else np.array(embeddings)
        labels = np.array(labels)

        if output_name:
            assert len(embeddings) == len(labels), "Lengths of embeddings and labels do not match"

            if add_name:
                data = [{'features': emb.tolist(), 'rating': lab, 'image_name': img_name, 'pred_resnet': pred.item()}
                        for emb, lab, img_name, pred in zip(embeddings, labels, image_names, predicts)]
            else:
                data = [{'features': emb.tolist(), 'rating': lab, 'pred_resnet': pred.item()} for emb, lab, pred in
                        zip(embeddings, labels, predicts)]

            with open(output_name, 'w', newline='') as csvfile:

                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

        return embeddings

    def _get_model_and_layer(self, model_name, layer, num_output_categories):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:

            model = Classifier(int(num_output_categories)).to(self.device)
            model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'))["model"])
            layer = model.backbone._modules.get('avgpool')
            return model,layer


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()