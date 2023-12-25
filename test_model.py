"""Resnet Testing Script

Usage:
    test_model.py --experiment EXPERIMENT_NAME

Options:
    --experiment Name of experiment as listed in constants.py [default: threecity2_resnet_multi_test1_boston]

"""

import warnings
import json
import os
import pandas as pd
import numpy as np
import json
import util
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage  # require scikit-image pkg
import sys

from PIL import Image
from torch.autograd import Variable
from torch import topk
from torch.nn import functional as F
from sklearn import svm
from sklearn.metrics import confusion_matrix
from docopt import docopt
from pprint import pprint
from os import walk
from sklearn.manifold import TSNE
from constants import *
from util import get_scores
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

def get_CAM_results(args):
    """
    Inputs:
        model_name: Path to trained model
        image_csv_path: Path to csv containing test image names (column: "image_name")
        image_directory: Path to folder containing test images
        num_output_labels: number of output classes for Resnet (2 for binary, 3 for multiclass)
    Outputs:
        data: Dataframe containing input image data from image_csv_path and column "pred_resnet" for class predicted by model.
    """
    model_name, num_classes, label_column, input_csv, image_directory = args['--model_name'], int(
        args['--num_classes']), args['--true_label'], args['--input_csv'], args['--image_dir']

    data = pd.read_csv(input_csv)
    data['top_act_mean'] = ''
    data['bottom_act_mean'] = ''
    data['top_act_std'] = ''
    data['bottom_act_std'] = ''
    data['total_act_std'] = ''
    count = 0

    def get_activation_stats(row):
        nonlocal count
        try:
            row.top_act_mean, row.bottom_act_mean, row.top_act_std, row.bottom_act_std, row.total_act_std = get_regional_activations(
                args, row.image_name, num_classes)

            # vec = img_2_vec.get_vec(img)
            #           row.features = json.dumps(vec.tolist())
            return row
        except Exception as e:
            print(e)

    data = data.apply(get_activation_stats, axis=1)
    return data


def get_misclassifications(args, data, class_markers=[0, 1, 2]):
    true_label = args['--true_label']
    misclassified = data.loc[data['pred_resnet'] != [class_markers[x - 1] for x in data[true_label]]]
    # misclassified = data.loc[data[true_label].apply(lambda x: int(x) - 1) - data['pred_resnet'].apply(lambda x: int(x)) != 0]
    # m = data.loc[data["trueskill_trash_category"].apply(lambda x: int(x) - 1) - data["pred_resnet"].apply(lambda x: int(x)) != 0]

    return (m)


def get_confusion_matrix(args, data):
    cm = confusion_matrix(data[args['--true_label']], data['pred_resnet'])
    return (cm)


def generate_CAM(args, image_name, num_classes, save_name=""):
    model_name, num_classes, label_column, input_csv, image_dir = args['--model_name'], int(args['--num_classes']), \
    args['--true_label'], args['--input_csv'], args['--image_dir']

    def get_activations(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]

    image = Image.open(image_dir + "/" + image_name)
    width, height = image.size
    img_2_vec = img_to_vec.Img2Vec(model=model_name, num_output_labels=num_classes)
    prediction_var = img_2_vec.process_image(image)

    model = img_2_vec.model
    final_layer = model._modules.get('layer4')
    activated_features = img_to_vec.SaveFeatures(final_layer)

    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()
    print("")

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    weight_softmax_params

    class_idx = topk(pred_probabilities, 1)[1].int()

    overlay = get_activations(activated_features.features, weight_softmax, class_idx)
    print(overlay)
    fig = plt.figure(figsize=(5, 4))
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(image)
    plt.imshow(overlay[0], alpha=0.5, cmap='jet')
    plt.imshow(skimage.transform.resize(overlay[0], (height, width)), alpha=0.3, cmap='jet');
    im_title = image_name + " | prediction: " + str(int(class_idx))
    plt.title(im_title)
    # plt.show()
    print(skimage.transform.resize(overlay[0], (height, width)))
    fig.tight_layout(pad=0)
    return (fig)


def get_regional_activations(args, image_name, num_classes, trash_class=0):
    model_name, num_classes, label_column, input_csv, image_dir = args['--model_name'], int(args['--num_classes']), \
    args['--true_label'], args['--input_csv'], args['--image_dir']

    def get_activations(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]

    image = Image.open(image_dir + "/" + image_name)
    width, height = image.size
    img_2_vec = img_to_vec.Img2Vec(model=model_name, num_output_labels=num_classes)
    prediction_var = img_2_vec.process_image(image)

    model = img_2_vec.model
    final_layer = model._modules.get('layer4')
    activated_features = img_to_vec.SaveFeatures(final_layer)

    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()
    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    weight_softmax_params
    
    class_idx = topk(pred_probabilities, 1)[1].int()

    overlay = get_activations(activated_features.features, weight_softmax, class_idx)
    top_activations = np.mean(skimage.transform.resize(overlay[0], (height, width))[:200])
    bottom_activations = np.mean(skimage.transform.resize(overlay[0], (height, width))[200:])
    total_std = np.std(skimage.transform.resize(overlay[0], (height, width)))
    top_std = np.std(skimage.transform.resize(overlay[0], (height, width))[:200])
    bottom_std = np.std(skimage.transform.resize(overlay[0], (height, width))[200:])
    print("total standard deviation of activations:", total_std)
    print("max weight:", np.max(skimage.transform.resize(overlay[0], (height, width))))
    print("min weight:", np.min(skimage.transform.resize(overlay[0], (height, width))))
    return (top_activations, bottom_activations, top_std, bottom_std, total_std)


def generate_CAM_forclass(args, image_name, num_classes, trash_class=0, save_name=""):
    model_name, num_classes, label_column, input_csv, image_dir = args['--model_name'], int(args['--num_classes']), \
    args['--true_label'], args['--input_csv'], args['--image_dir']

    def get_activations(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]

    image = Image.open(image_dir + "/" + image_name)
    width, height = image.size
    img_2_vec = img_to_vec.Img2Vec(model=model_name, num_output_labels=num_classes)
    prediction_var = img_2_vec.process_image(image)

    model = img_2_vec.model
    final_layer = model._modules.get('layer4')
    activated_features = img_to_vec.SaveFeatures(final_layer)

    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    weight_softmax_params
    class_idx = topk(pred_probabilities, 1)[1].int()
    print("Predicted trash class for " + image_name + " is " + str(int(class_idx)))
    overlay = get_activations(activated_features.features, weight_softmax, trash_class)
    fig = plt.figure(figsize=(5, 4))
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(image)
    plt.imshow(overlay[0], alpha=0.5, cmap='jet')
    plt.imshow(skimage.transform.resize(overlay[0], (height, width)), alpha=0.3, cmap='jet');
    im_title = image_name + " | prediction: " + str(int(class_idx)) + " | activation for: " + str(int(trash_class))
    plt.title(im_title)
    # plt.show()
    print(prediction)
    fig.tight_layout(pad=0)
    return (fig)


def features_to_array(features_string):
    def to_array(list_of_lists):
        return np.array([np.array(xi) for xi in list_of_lists])

    features = []
    for i, row in features_string.iterrows():
        if type(row.features) != str:
            print(row)
        feat = json.loads(row.features)
        features.append(feat)

    features = to_array(features)
    return features


def analyze_results(test_results, title, city, col_true="rating", col_pred="pred_resnet", binary_threshold=-1):
    print("Confusion Matrix,", title)
    m = test_results.loc[test_results[col_true] - test_results[col_pred] != 0]
    accuracy = 1 - len(m) / len(test_results)
    print(accuracy * 100, "% accuracy")
    print(len(m), " misclassified out of ", len(test_results), " total images")
    print(confusion_matrix(test_results[[col_true]], test_results[[col_pred]]))
    print(test_results[test_results[col_true] == test_results[col_pred]].groupby(col_true).count()["image_name"] / (
    test_results.groupby(col_true).count()["image_name"]))
    print("F1 score: ", f1_score(test_results[col_true], test_results[col_pred], average='weighted'))
    trueskill_trash = test_results[test_results["score"] > 0]
    fig = plt.figure(figsize=(5, 4))
    plt.scatter(trueskill_trash[["score"]], trueskill_trash[[col_pred]])
    plt.xlabel("Trueskill score")
    plt.ylabel("Model prediction")
    plt.title(title)
    plt.yticks([0, 1, 2, 3])
    for threshold in THRESHOLDS[city]:
        plt.axvline(x=threshold, color="black", linestyle="--")
    if binary_threshold >= 0 and binary_threshold < 4:
        plt.axvline(x=thresholds[city][binary_threshold], color="black", linestyle="-")
    return (fig)


def analyze_distribution(data, city, save_name="distribution.png", title=None):
    """
    Inputs:
    - data: dataframe with columns
                                    "score" (trueskill score, float)
                                    "image_name" (name of image jpg)
                                    "rating" (true trash class label)
    - city: ["boston", "la", "detroit"]
    - title: string title for plot
    """
    plt.figure(figsize=(5, 8))
    plt.subplot(2, 1, 1)
    plt.hist(data["score"], bins=np.arange(min(data[data["score"] != 0]["score"]), max(data["score"]) + 1, 1))
    plt.title(title)
    for threshold in thresholds[city]:
        plt.axvline(x=threshold, color="black", linestyle="--")
    plt.subplot(2, 1, 2)
    plt.hist(data["score"].apply(lambda x: 0 if x == 0 else 1))
    plt.xticks([0, 1])
    plt.title("number of images labelled as no trash")
    plt.subplots_adjust(hspace=.5)
    plt.savefig(save_name)
    plt.close()
    distribution = data.groupby("rating").count()["image_name"]
    print("Distribution of data by category")
    print(distribution)
    return (distribution)


def plot_TSNE(features_data, title, cmap, save_name="tsneplot.png", p=30):
    features = features_to_array(pd.DataFrame(features_data.features))
    print("done")
    true_label = features_data[cmap]
    X_embedded = TSNE(n_components=2, perplexity=p).fit_transform(features)
    plt.scatter(pd.DataFrame(X_embedded)[0], pd.DataFrame(X_embedded)[1], c=true_label, alpha=.5)
    plt.title(title)
    plt.colorbar()
    plt.set_cmap('jet')
    plt.savefig(save_name)


def make_CAM_images(args, data, dir_name="error_analysis/cams/", count=50):
    print(len(data), "data points")
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " created ")
    else:
        print("Directory ", dir_name, " exists ")

    for i, row in data[:min(len(data), count)].iterrows():
        img_cam = generate_CAM(args, row.image_name, args["--num_classes"])
        img_cam.savefig(dir_name + row.image_name)

        img_cam.canvas.draw()


if __name__ == "__main__":

    experiment = args['EXPERIMENT_NAME']
    test_args = experiments[experiment]
    predictions = get_predictions(test_args)
    dir_name = "error_analysis/" + experiment + "/"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    predictions_scores = get_scores(predictions, "data/trueskill_" + test_args["--test_city"] + ".csv")
    predictions_scores.to_csv(dir_name + experiment + ".csv")
    fig_results = analyze_results(predictions_scores, experiment, test_args["--test_city"], "rating", "pred_resnet",
                                  binary_threshold=1)
    fig_results.savefig("error_analysis/" + experiment + "/" + "results.png")

    # Generate CAMs
    for r in range(0, args["--num_classes"]):
        sub_class = predictions[predictions["rating"] != predictions["pred_resnet"]]
        sub_class = sub_class[sub_class["rating"] == r]
        dir_name = "error_analysis/" + experiment + "/cams/incorrect_" + str(r) + "/"
        print(dir_name)
        make_CAM_images(test_args, sub_class, str(dir_name), 30)
    for r in range(0, args["--num_classes"]):
        sub_class = predictions[predictions["rating"] == predictions["pred_resnet"]]
        sub_class = sub_class[sub_class["rating"] == r]
        dir_name = "error_analysis/" + experiment + "/cams/correct_" + str(r) + "/"
        print(dir_name)
        make_CAM_images(test_args, sub_class, str(dir_name), 30)
