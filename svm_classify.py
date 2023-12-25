"""  Train classifier from extracted image vectors or to test trained model and store results in existing features csv

Usage:
    svm_classify.py   --features_csv_path CSV_PATH --svmodel_path MODEL_PATH
                        [--test TEST] [--var_type VAR_TYPE]


Options:
    --features_csv_path Path to csv containing image names and features from extract_vectors.py
    --test Bool for whether to train new model (False) or test existing model (True) [default: True]
    --svmodel_path: Path to trained model(test=True) /where to store new trained model (test=False)
    --var_type 'discrete' for SVC, 'continuous' for SVR [default: 'discrete']
"""

#example: python svm_classify.py --features_csv_path ./temp.csv --svmodel_path ./svm_temp --test False --var_type 'discrete'

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import warnings
import pickle
from util import print_to_console_and_file
from docopt import docopt
warnings.filterwarnings("ignore")



def features_to_array(features_string):
    """
    Args:
        features_string: df with features as strings

    Returns:
        Converts string list of features to an array of floats for every row

    """
    def to_array(list_of_lists):
        return np.array([np.array(xi) for xi in list_of_lists])

    features = []
    for i, row in features_string.iterrows():
        feat = json.loads(row.features)
        features.append(feat)
    features = to_array(features)
    return features


def train_model(training_csv_path, model_save_name):
    """
    Args:
        training_csv_path: path for csv containing vectorized training images
        model_save_name: path for saving SVC model
    Returns:
        trained SVC model

    """
    data = pd.read_csv(training_csv_path)
    model = SVC(class_weight='balanced')
    input_features = features_to_array(pd.DataFrame(data.features))
    output_labels = pd.DataFrame(data.rating)
    model.fit(input_features, output_labels)

    with open(model_save_name, 'wb') as fid:
        pickle.dump(model, fid)
    return model

def train_svr_model(training_csv_path, model_save_name):
    """
    Args:
        training_csv_path: path to csv with vectorized training images
        model_save_name: path to save trained svr model

    Returns:
        svr model also saved utilizing pickle at path given in parameters
    """

    data = pd.read_csv(training_csv_path)
    model = SVR()
    input_features = features_to_array(pd.DataFrame(data.features))
    output_labels = pd.DataFrame(data.rating).astype('int64')
    model.fit(input_features, output_labels)
    print("score: ", model.score(input_features, output_labels))
    with open(model_save_name, 'wb') as fid:
        pickle.dump(model, fid)
    return(model)


def predict_test(test_path, clf, output_path, output_col="pred_svm"):
    """

    Args:
        test_path: path of csv containing vectorized test images
        clf: the loaded trained svm classifier
        output_path: output csv path to save the output
        output_col: column name to save the svm predictions

    Returns:
        outputs the df that has the added svm predictions

    """

    test_data = pd.read_csv(test_path)
    test_predictions = clf.predict(features_to_array(test_data))
    test_data[output_col] = test_predictions
    test_data.to_csv(output_path, index=False)
    return test_data


if __name__=="__main__":
    args = docopt(__doc__)
    print(args)
    csv_path = args['--features_csv_path']
    model_path = args['--svmodel_path']
    test = args['--test'] == "True"
    var_type = args['--var_type']

    if test:
        classifier = pickle.load(open(model_path, 'rb'), encoding='latin1')
        testing = predict_test(csv_path, classifier, "output_test_SVC.csv")
    else:
        if var_type == 'continuous':
            classifier = train_svr_model(csv_path, model_path)
        else:
            classifier = train_model(csv_path, model_path)
