"""

Creates CSV of true labels given CSV of thresholds


--output: output name of the CSV created
--input: input of CSV with raw trueskill scores
--city: the city that the trueskill scores comes from

example use:

 python3 discretize_trueskill.py --output labelled_data/detroit2/detroit_multi_val_old.csv --input combined_trueskill.csv --city detroit

"""

usage = """ 
Creates csv using inputted thresholds and raw trueskill scores to produce true labels.

Usage:
    discretize_trueskill.py --output <output> --input <input> --city <city>

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
from scipy import stats
from constants import *
from util import get_scores


def get_rating (score, thresh):
    """Given a trueskill score and list of thresholds, returns corresponding rating"""
    rating = 0
    for i in range(0, len(thresh)):
        if score >= thresh[i]:
            rating = i + 1
    return rating

def trueskill_to_rating (data, thresh, rating_col="rating"):
    """Given dataset containing trueskill scores, add column containing corresponding ratings"""
    data[rating_col] = data["score"].apply(lambda x: get_rating(x, thresh))
    return data



def get_trueskill_threshold(rating, model):
    """calculates trueskill value for associated RA value according to OLS"""
    coef = model.coef_
    intercept = model.intercept_
    trueskill = (rating - intercept)/coef
    return trueskill


def fit_OLS(x, y):
    """Returns ordinary least squares model fit to x and y data"""
    reg_model = LinearRegression()
    reg_model.fit(x, y)
    print ("OLS regression: RA rating = ", reg_model.coef_, "* (trueskill) + ", reg_model.intercept_)
    print(get_trueskill_threshold(1.5, reg_model), get_trueskill_threshold(2, reg_model),get_trueskill_threshold(2.5, reg_model),get_trueskill_threshold(3, reg_model))
    return reg_model


def add_labels(trueskill_data, dataset, cutoff1, cutoff2, cutoff3=-1):
    """Based on threshold inputs and trueskill data, creates a new file with discretized labels"""
    if cutoff3 == -1:
        cutoff3 = cutoff2
    conditions = [
        (trueskill_data <= cutoff1),
        (trueskill_data <= cutoff2),
        (trueskill_data > cutoff3)
    ]
    #categories = ["0- no trash", "1- some trash", "2- lots of trash"]
    categories = [1, 2, 3]
    trash_class_data["rating"] = np.select(conditions, categories)
    trash_class_data.to_csv("labels_" + file_path, index = False)

"""Given an array of trueskill data and a single trueskill value, returns the percentile of value"""
def get_percentile(arr, value):
    return (stats.percentileofscore(arr, value))

if __name__ == "__main__":

    args = docopt(usage)
    print(args)  

    data_file =  args["<output>"] 
    trueskill_file = args["<input>"]
    city = args["<city>"]
    thresholds = THRESHOLDS[city]

    discretized_data = get_scores(data_file, trueskill_file)
    discretized_data = trueskill_to_rating(discretized_data, thresholds)
    print(data_file.groupby(["rating"]).count()["score"])
    discretized_data.to_csv(data_file)