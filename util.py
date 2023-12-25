import pandas as pd
import os

from shutil import copyfile
from os.path import join as join_paths
from os.path import isdir as dir_exists
from os import makedirs as make_folder



def add_prefix_file(file_path, prefix):
    split_path = file_path.rsplit('/', 1)
    if len(split_path) != 2:
        return prefix + '_' + file_path
    return ('/' + prefix + '_').join(split_path)


def add_suffix_file(file_path, suffix):
    split_path = file_path.rsplit('.', 1)
    return ('_' + suffix + '.').join(split_path)


def copy_all_images(list_images, src, dst):
    src_files = os.listdir(src)
    print('Attempting to copy {} images from {} to {}'.format(len(list_images), src, dst))
    src_files_set = set(src_files)
    id = 0
    for image in list_images:
        if image in src_files_set:
            copyfile(join_paths(src, image), join_paths(dst, str(id) + "_" + image))
            id += 1
        else:
            print(image, "Not Found...")
    print("Found {} images, copied from {} to {}".format(id , src, dst))

def safe_create_dir(directory):
    if dir_exists(directory):
        print('{} already exists'.format(directory))
    else:
        make_folder(directory)
        return True


def add_trash_category(data, title, threshold):
    data[title] = (data.trueskill_trash_score >= threshold).astype(int)
    return data


def add_three_categories(data, title, level_2_threshold, level_3_threshold):
    data[title] = -1

    def add_category(row):
        if row.trueskill_trash_score < level_2_threshold:
            row.trueskill_trash_category = 1
        elif row.trueskill_trash_score < level_3_threshold:
            row.trueskill_trash_category = 2
        else:
            row.trueskill_trash_category = 3
        return row

    data = data.apply(add_category, axis=1)
    return data


def print_to_console_and_file(text,file):
    if file:
        file.write(text + '\n')
    print(text)


def split_into_cities(data, prefix):
    groups = data.groupby('city')
    for city, group in groups:
        group.to_csv('{}_{}.csv'.format(prefix,city), index=False)

def get_scores(results_csv, trueskill_file, pred_col="score"):
    """Given a csv containing image_names, and the relevant city, appends score column
    containing corresponding trueskill score for each image"""
    if "score" in results_csv.columns:
        print("trueskill score already in file")
        return results_csv
    else:
        trueskill = pd.read_csv(trueskill_file)
        print("Number of images in file: ", len(results_csv))
        print("Size of trueskill data: ", len(trueskill))
        trueskill_unique = trueskill.drop_duplicates(subset=["image_name"])
        print("Size of unique trueskill data: ", len(trueskill_unique))
        results_csv["image_name"] = results_csv["image_name"].apply(extract_img_name)
        print(len(results_csv))
        results_with_score = pd.merge(results_csv, trueskill_unique[["image_name", "score"]], on="image_name")
        print("Number of images with trueskill: ", len(results_with_score))
        if (len(results_csv) != len(results_with_score)):
            results_without_score = pd.concat([results_with_score[["image_name"]], results_csv[["image_name"]]]).drop_duplicates(keep=False)
            results_without_score["score"] = 0
            results_without_score = pd.merge(results_without_score, results_csv, on="image_name")
            results_total = pd.concat([results_with_score, results_without_score])
            print("Total size of merged data: ", len(results_total))
        else:
            results_total = results_with_score
        if (len(results_total) == len(results_csv)):
            return results_total
        else:
            print("could not merge- size mismatch")
            return


def extract_img_name(img_name):
    """
    Args:
        img_name: Saved image name in csv

    Returns:
        image name with parsed out id added in copy all images util function
    """
    parts = img_name.split("_", 1)
    if len(parts) >= 2:
        return parts[1]
    return image_name



def get_splits_csv(dir, labelled_data=""):
    imgs=[]
    ratings=[]
    splits=[]

    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            if name[-4:]=='.jpg':
                file_name=root.replace(dir,'')
                imgs.append(name)
                ratings.append(int(file_name[-1:]))
                splits.append(file_name[1:-2])
    folder_data = pd.DataFrame({"image_name":imgs, "rating":ratings, "split":splits})

    if len(labelled_data) > 0:
            folder_data = pd.merge(folder_data, pd.read_csv(labelled_data)[["image_name", "score"]], on="image_name")
    print(folder_data)
    return(folder_data[folder_data["split"]=="train"], folder_data[folder_data["split"]=="val"])