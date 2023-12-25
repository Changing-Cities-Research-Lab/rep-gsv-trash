"""GLOBAL CONSTANTS"""
from __future__ import absolute_import

THRESHOLDS = {"boston":[23.253, 28.9486, 36.8692], "la": [21.9734, 26.95145, 32.67290], "detroit": [19.06599, 23.99331, 31.47627]}
PERCENTAGE_VALIDATION = .1
CITIES = ["boston", "detroit", "la"]

#holds parameters for different experiments:
#model_name is where to store the trained resnet model
#image_dir is the directory where you can find all the images
#input_csv the csv with the original data to run the pipeline
#true_label what the true label column will be called in the csv
#num_classes the number of classes to classify over
#test_city the city 


EXPERIMENTS = {
    "threecity2_resnet_binary_test1_boston": {
        "--model_name" : "models/threecity2_binary_model",
        "--image_dir"  : "data/boston/all_images",
        "--input_csv"  : "labelled_data/boston/boston_binary_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "boston"
    },
    "threecity2_resnet_binary_test1_detroit2": {
        "--model_name" : "models/threecity2_binary_model",
        "--image_dir"  : "data/detroit_trash/all_images",
        "--input_csv"  : "labelled_data/detroit2/detroit_binary_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2
    },
    "threecity2_resnet_binary_test1_la": {
        "--model_name" : "models/threecity2_binary_model",
        "--image_dir"  : "data/la_trash/all_images",
        "--input_csv"  : "labelled_data/la/la_binary_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2
    },
    "threecity2_resnet_lowhigh_test1_detroit2": {
        "--model_name" : "models/threecity2_lowhi_model",
        "--image_dir"  : "data/detroit_trash/all_images",
        "--input_csv"  : "labelled_data/detroit2/detroit_lowhigh_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2
    },
    "threecity2_resnet_lowhigh_test1_boston": {
        "--model_name" : "models/threecity2_lowhi_model",
        "--image_dir"  : "data/boston/all_images",
        "--input_csv"  : "labelled_data/boston/boston_lowhigh_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2
    },
    "threecity2_resnet_lowhigh_test1_la": {
        "--model_name" : "models/threecity2_lowhi_model",
        "--image_dir"  : "data/la_trash/all_images",
        "--input_csv"  : "labelled_data/la/la_lowhigh_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2
    },
    "threecity2_resnet_multi_test1_boston": {
        "--model_name" : "models/threecity2_multi_model",
        "--image_dir"  : "data/boston/all_images",
        "--input_csv"  : "labelled_data/boston/boston_multi_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 4
    },
    "threecity2_resnet_multi_test1_detroit2": {
        "--model_name" : "models/threecity2_multi_model",
        "--image_dir"  : "data/detroit_trash/all_images",
        "--input_csv"  : "labelled_data/detroit2/detroit_multi_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 4
    },
    "threecity2_resnet_multi_test1_la": {
        "--model_name" : "models/threecity2_multi_model",
        "--image_dir"  : "data/la_trash/all_images",
        "--input_csv"  : "labelled_data/la/la_multi_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 4
    },
    "threecity2_resnet_high_test1_boston": {
        "--test_city"  : "boston",
        "--model_name" : "models/threecity2_high_binary_model",
        "--image_dir"  : "data/boston/all_images",
        "--input_csv"  : "labelled_data/boston/boston_multi_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2
    },
    "threecity2_resnet_high_test1_detroit": {
        "--test_city"  : "detroit",
        "--model_name" : "models/threecity2_high_binary_model",
        "--image_dir"  : "data/detroit_trash/all_images",
        "--input_csv"  : "labelled_data/detroit2/detroit_multi_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2
    },
    "threecity2_resnet_high_test1_la": {
        "--test_city"  : "la",
        "--model_name" : "models/threecity2_high_binary_model",
        "--image_dir"  : "data/la_trash/all_images",
        "--input_csv"  : "labelled_data/la/la_multi_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2
    },
    "final_resnet_binary_boston": {
        "--model_name" : "models/combined_binary_model_FINAL",
        "--image_dir"  : "data/boston/all_images",
        "--input_csv"  : "labelled_data/boston2/boston_binary_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "boston"
    },
    "final_resnet_binary_detroit": {
        "--model_name" : "models/combined_binary_model_FINAL",
        "--image_dir"  : "data/detroit_trash/all_images",
        "--input_csv"  : "labelled_data/detroit2/detroit_binary_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "detroit"
    },
    "resampled_resnet_binary_detroit": {
        "--model_name" : "models/resampled_binary_model_FINAL",
        "--image_dir"  : "data/detroit_trash/all_images",
        "--input_csv"  : "labelled_data/detroit2/detroit_binary_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "detroit"
    },
    "resampled_resnet_binary_boston": {
        "--model_name" : "models/resampled_binary_model_FINAL",
        "--image_dir"  : "data/boston/all_images",
        "--input_csv"  : "labelled_data/boston2/boston_binary_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "boston"
        },
    "resampled_resnet_binary_la": {
        "--model_name" : "models/resampled_binary_model_FINAL",
        "--image_dir"  : "data/la_trash/all_images",
        "--input_csv"  : "labelled_data/la/la_binary_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "la"
        },
    "resampled_resnet_lowhigh_detroit": {
        "--model_name" : "models/resampled_lowhigh_model_FINAL",
        "--image_dir"  : "data/detroit_trash/all_images",
        "--input_csv"  : "labelled_data/detroit2/detroit_lowhigh_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "detroit"
    },
    "resampled_resnet_lowhigh_boston": {
        "--model_name" : "models/resampled_lowhigh_model_FINAL",
        "--image_dir"  : "data/boston/all_images",
        "--input_csv"  : "labelled_data/boston2/boston_lowhigh_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "boston"
        },
    "resampled_resnet_lowhigh_la": {
        "--model_name" : "models/resampled_lowhigh_model_FINAL",
        "--image_dir"  : "data/la_trash/all_images",
        "--input_csv"  : "labelled_data/la/la_lowhigh_val_old.csv",
        "--true_label" : "rating",
        "--num_classes": 2,
        "--test_city": "la"
        }
}