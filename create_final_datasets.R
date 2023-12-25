# -------------------------------------------------------------------------
# Created by: Jackie Hwang                          
# Date created: Oct 07, 2022                
# Last revised: Apr 07, 2023               
# Project: GSV          
# Subproject: GSV trash replication repo  
# Re: Create final trash dataset for public use      
# -------------------------------------------------------------------------

# IMPORTANT: DO NOT INCLUDE THIS SCRIPT IN FINAL PUBLIC REPO ####

# Script Description ------------------------------------------------------

# This script compiles all of the raw prediction results into a single dataset 
# (seperate datasets for "recent" and "time series"), final training data with labels, 
# and ACS data and moves them into a replication-repo folder. 

# IMPORTANT NOTE: #### 
# These datasets will need to be updated after trash predictions are completed 
# for the fixed time series images and when parcels with missing geocodes are fixed.

# Inputs:
# Recent and time Series data for each city
# training data from MTurk/Trueskill results
# ACS data for external validity

# Outputs:
# .csv files of raw data for recent + time series

# Update log: 
# 01/20/23: creating final datasets for all cities
# 01/25/23: create block face IDs, fix missing geocodes
# 02/01/23: create ACS data for external validity, add MTurk training data
# 02/07/23: get parcels with missing geocodes
# 02/15/23: add MTurk vs. Coding sessions reliability data, fix export census/acs data
# 04/03/23: clean cross validation results
# 04/07/23: change multiclass svm predictions to categories

# Setup -------------------------------------------------------------------

# Packages: 
library('tidyverse') #v1.3.2
library('foreach') #v1.5.2

# Directories: 
homedir <- 'G:/My Drive/Changing Cities Research Lab/PROJECT FOLDER  GSV/'
workdir <- 'machine learning/ml-trash/final results/'
savedir <- paste0(workdir, "replication_repo/")
gitdir <- paste0(dirname(rstudioapi::getActiveDocumentContext()$path), "/")

# Import data: 
setwd(paste0(homedir, workdir))

# Recent data for each city
dat_bos_recent <- read_csv("boston_trash_recent.csv")
dat_det_recent <- read_csv("detroit_trash_recent.csv")
dat_la_recent <- read_csv("la_trash_recent.csv")

# austin and philly recent = most recent year in time series
dat_aus_recent <- read_csv("austin_trash_time_series.csv")
dat_phl_recent <- read_csv("philadelphia_trash_time_series.csv")

# Run loop to create year column and get most recent year
dat <- list(dat_aus_recent, dat_phl_recent)
foreach (i = 1:length(dat)) %do% {
  pred <- dat[[i]]
  names(pred)[1] <- "image_name"
  pred$year <- trimws(sapply(strsplit(pred$image_name, "_"), `[[`, 2))
  sub("^.*_\\s+(.*?)\\s+_.*$", "\\1", pred$image_name)
  
  #get most recent year
  pred <- pred[order(pred$parcel_id,pred$year),]  # Sort by ID and year
  pred <- pred[!duplicated(pred$parcel_id, fromLast=T),] # Keep last observation per ID
  dat[[i]] <- pred
  return(NULL)
}
dat_aus_recent <- dat[[1]]
dat_phl_recent <- dat[[2]]
rm(dat, pred, i)

# time series data for each city
# LA TS data is incomplete - based off the old set where 1/2 not uploaded
dat_bos_ts <- read_csv("boston_trash_time_series.csv")
dat_det_ts <- read_csv("detroit_trash_time_series.csv")
dat_la_ts <- read_csv("la_trash_time_series/la_trash_time_series_results_with_tract.csv")
dat_aus_ts <- read_csv("austin_trash_time_series.csv")
dat_phl_ts <- read_csv("philadelphia_trash_time_series.csv")

# Census/ACS data
censusacs <- 
  read_csv(file = "C:/Users/jacks17/Dropbox/0Non-Working/2REFERENCE/1RESEARCH/Datasets/US Census and American Community Survey/Aggregated Longitudinal Datasets/LTDB-ACS Merge Archive/ltdb7010_acs1317_2010boundaries_merge_clean.csv")

# Training data
dat_train <- 
  read_csv(file = paste0(homedir, "machine learning/ml-trash/official-train-and-test-sets/training_multi_full.csv"))

# Reliability data 
dat_reliability_pairs <- 
  read_csv(file = paste0(homedir, "background and drafts/Drafts/reliability/replication-data/pairs.csv"),
          # fix importing for some values
            col_types = list(
             pairs_cs_choice2 = col_double(),
             pairs_cs_choice3 = col_double(),
             pairs_cs_choice4 = col_double(),
             pairs_mturk_choice2 = col_double(),
             pairs_mturk_choice3 = col_double(),
             pairs_mturk_choice4 = col_double()
           ))
dat_reliability_single_image <- 
  read_csv(file = paste0(homedir, "background and drafts/Drafts/reliability/replication-data/single_image.csv"),
           # fix importing for some values
           col_types = list(
             pred_svm = col_double(),
             single_cs_rating5 = col_double(),
             single_cs_rating6 = col_double()
           ))

# Cross-validation results
dat_crossval_pa <- # presence/absence resulst
  foreach (i = c(0:9)) %do% {
    out <- read_csv(file = paste0(homedir, workdir, "cross_val_results/pa/", i, "/val_preds.csv"))
    # only keep necessary variables
    out <- out %>% select(city, image_name, rating, score, rating_lh, rating_multi, pred_svm)
    return(out)
  }
names(dat_crossval_pa) <- c(1:10)

dat_crossval_lh <- # low/high results
  foreach (i = c(0:9)) %do% {
    out <- read_csv(file = paste0(homedir, workdir, "cross_val_results/lh/", i, "/val_preds.csv"))
    # only keep necessary variables
    out <- out %>% select(city, image_name, rating, score, rating_lh, rating_multi, pred_svm)
    return(out)
  }
names(dat_crossval_lh) <- c(1:10)

dat_crossval_multi <- # multiclass results
  foreach (i = c(0:9)) %do% {
    out <- read_csv(file = paste0(homedir, workdir, "cross_val_results/multi/", i, "/val_preds.csv"))
    # only keep necessary variables
    out <- out %>% select(city, image_name, rating, score, rating_lh, rating_multi, pred_svm)
    # categorize svm predictions
    out <- 
      out %>% 
      mutate(pred_svm = ifelse(pred_svm <.5, 0, ifelse(pred_svm <1.5, 1, 2)))
    return(out)
  }
names(dat_crossval_multi) <- c(1:10)

# Parameters:
cities <- c("austin", "boston", "detroit", "la", "philly")

# Main Script -------------------------------------------------------------

# ML Model Performance - Cross-Validation Results #-----------------------------

# combine each set of results into single files ####
foreach (i = 1:length(dat_crossval_pa)) %do% {
  # add index for cross-validation set 
  dat_crossval_pa[[i]] <- 
    dat_crossval_pa[[i]] %>%
    mutate(crossval_set = i)
  return(NULL)
  }
dat_crossval_pa <- do.call("rbind", dat_crossval_pa)

foreach (i = 1:length(dat_crossval_lh)) %do% {
  # add index for cross-validation set 
  dat_crossval_lh[[i]] <- 
    dat_crossval_lh[[i]] %>%
    mutate(crossval_set = i)
  return(NULL)
}
dat_crossval_lh <- do.call("rbind", dat_crossval_lh)

foreach (i = 1:length(dat_crossval_multi)) %do% {
  # add index for cross-validation set 
  dat_crossval_multi[[i]] <- 
    dat_crossval_multi[[i]] %>%
    mutate(crossval_set = i)
  return(NULL)
}
dat_crossval_multi <- do.call("rbind", dat_crossval_multi)

# Final ML Trash Results #------------------------------------------------------

# make each dataset uniform format ####
# Note: LA data - remove invalid images, change stage2_good = -1 to 1

# recent datasets ####

# austin ##

# aus_parcels <-
#   read_csv("G:/My Drive/Changing Cities Research Lab/PROJECT FOLDER  GSV/gsv-images/full_sets/Austin Scraping/austin_parcels_2016.csv")
# austin parcel data does not have lat-lon/geocode data to merge for parcels with missing ids (n = 554)

dat_aus_recent <- 
  dat_aus_recent %>%
  # add city variable
  mutate(
    city = "austin", 
    # fix parcel ids with missing geocodes that need lead 0
    parcel_id = ifelse(is.na(tract), sprintf("%010.0f", parcel_id), as.character(parcel_id))) %>%
    # convert
  # keep only variables needed
  select(
    city, image_name, parcel_id, address, latitude, longitude, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm) %>%
  # rename variables to standardize across datasets
  rename(
    lat = latitude, 
    lon = longitude
  )

# boston ##

# add lat/lon data 
bos_parcels <- 
  read_csv("G:/My Drive/Changing Cities Research Lab/PROJECT FOLDER  GSV/gsv-images/full_sets/Boston Scraping/Parcels 2015 BARI CSV.csv")
# remove duplicated parcel_nums
bos_parcels <- 
  unique(bos_parcels %>% select(parcel_num, address, X, Y, Blk_ID_10, BG_ID_10, CT_ID_10))
bos_parcels <- 
  bos_parcels[!duplicated(bos_parcels$parcel_num), ]
# merge with boston data
dat_bos_recent <- 
  left_join(
    dat_bos_recent, 
    bos_parcels %>% select(parcel_num, X, Y), 
    by = c("parcel_id" = "parcel_num")
  ) %>% 
  # replace lat and lon variables and change to string
  mutate(
    lat = as.character(Y), 
    lon = as.character(X)
  )

dat_bos_recent <- 
  dat_bos_recent %>%
  # add city variable
  mutate(
    city = "boston", 
    parcel_id = as.character(parcel_id)
  ) %>%
  # keep only variables needed
  select(
    city, image_name, parcel_id, address, lat, lon, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm)
  # rename variables to standardize across datasets - n/a

# detroit ##

dat_det_recent <- 
  dat_det_recent %>%
  # add city variable
  mutate(
    city = "detroit", 
    parcel_id = as.character(parcel_id)
  ) %>%
  # keep only variables needed
  select(
    city, image_name, parcel_id, address, latitude, longitude, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm) %>% 
  # rename variables to standardize across datasets 
  rename(
    lat = latitude, 
    lon = longitude
  )

# la ##

# la parcel data does not have census geocode data to merge for parcels with missing info (n = 91)
# add lat/lon data when missing
la_parcels <- read_csv("G:/My Drive/Changing Cities Research Lab/PROJECT FOLDER  GSV/gsv-images/full_sets/LA Scraping/LA_Parcels.csv")
# keep variables needed
la_parcels <- 
  unique(la_parcels %>% select(AIN, PropertyLocation, CENTER_LAT, CENTER_LON))

# merge with la data
dat_la_recent <- 
  left_join(
    dat_la_recent, 
    la_parcels %>% select(AIN, CENTER_LAT, CENTER_LON), 
    by = c("parcel_id" = "AIN")
  ) %>% 
  # replace lat and lon variables and change to string when missing
  mutate(
    lat = ifelse(lat == "None", as.character(CENTER_LAT), lat),
    lon = ifelse(lon == "None", as.character(CENTER_LON), lon)
  ) %>% 
  # remove additional variables
  select(-CENTER_LAT, -CENTER_LON)

# add la geocodes when missing if available 
la_geo <- read_csv("G:/My Drive/Changing Cities Research Lab/PROJECT FOLDER  GSV/gsv-images/full_sets/LA Scraping/la_census_info_updated.csv")
# reformat data to merge
la_geo <- 
  la_geo %>%
  mutate(
    block = as.character(block), 
    block = substr(block, nchar(block) - 3, nchar(block)),
    blkgrp = as.numeric(substr(block, 1, 1)), 
    block = as.numeric(block)
  ) %>% 
  rename(
    block_add = block, 
    blkgrp_add = blkgrp, 
    tract_add = tract
  )

# merge with la data
dat_la_recent <- 
  left_join(
    dat_la_recent, 
    la_geo
  ) %>% 
  mutate(
    tract = ifelse(tract < 1, tract_add, tract), 
    blkgrp = ifelse(blkgrp < 1, blkgrp_add, blkgrp), 
    block = ifelse(block < 1, block_add, block)
  )

dat_la_recent <- 
  dat_la_recent %>%
  # remove invalid data
  filter(invalid != 1) %>% #n=9539 images
  mutate(
    # change -1 in stage2_good to 1
    stage2_good = ifelse(stage2_good == -1, 1, stage2_good), #n=2386
    # change pred_svm to NA if -1
    pred_svm = ifelse(pred_svm == -1, NA, pred_svm)
  ) %>%
  # add city variable
  mutate(
    city = "la", 
    parcel_id = as.character(parcel_id)
  ) %>%
  # keep only variables needed
  select(
    city, image_name, parcel_id, address, lat, lon, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm)


# philly ## 

dat_phl_recent <- 
  dat_phl_recent %>%
  # add city variable
  mutate(
    city = "philly", 
    parcel_id = as.character(parcel_id), 
    lat = as.character(lat), 
    lon = as.character(lon)
  ) %>%
  # keep only variables needed
  select(
    city, image_name, parcel_id, address, lat, lon, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm)
  # rename variables to standardize across datasets - n/a


# time series datasets ####

# austin
# austin has 554 parcels with missing geocode data due to parcel_ids missing leading 0 & austin parcel data does not have lat-lon 
dat_aus_ts <- 
  dat_aus_ts %>%
  mutate(
    # add city variable
    city = "austin",  
    # fix parcel ids with missing geocodes that need lead 0
    parcel_id = ifelse(is.na(tract), sprintf("%010.0f", parcel_id), as.character(parcel_id)),
    # get year variable from image name
    year = trimws(sapply(strsplit(image_name, "_"), `[[`, 2)),
    year = sub("^.*_\\s+(.*?)\\s+_.*$", "\\1", year)
  ) %>%
  # keep only variables needed
  select(
    city, year, image_name, parcel_id, address, latitude, longitude, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm) %>%
  # rename variables to standardize across datasets
  rename(
    lat = latitude, 
    lon = longitude
  )

# boston
# merge lat-long data
dat_bos_ts <- 
  left_join(
    dat_bos_ts, 
    bos_parcels %>% 
      select(parcel_num, X, Y, address, Blk_ID_10) %>% 
      rename(address_parcels = address) %>% # create separate address variable
      mutate(
        tract_parcels = as.numeric(substr(as.character(Blk_ID_10), 6, 11)), 
        blkgrp_parcels = as.numeric(substr(as.character(Blk_ID_10), 12, 12)), 
        block_parcels = as.numeric(substr(as.character(Blk_ID_10), 12, 15))
      ), 
    by = c("parcel_id" = "parcel_num")
  ) %>% 
  # replace lat and lon variables and change to string
  mutate(
    lat = as.character(Y), 
    lon = as.character(X)
  ) %>% 
  # add address and census info if missing
  mutate(
    address = ifelse(is.na(address), address_parcels, address), 
    tract = ifelse(is.na(address), tract_parcels, tract), 
    blkgrp = ifelse(is.na(address), blkgrp_parcels, blkgrp), 
    block = ifelse(is.na(address), block_parcels, block)
  ) %>%
  # remove additional variables
  select(-address_parcels, -Blk_ID_10, -tract_parcels, -blkgrp_parcels, -block_parcels)
rm(bos_parcels)

dat_bos_ts <- 
  dat_bos_ts %>% 
  mutate(
    # add city variable
    city = "boston", 
    parcel_id = as.character(parcel_id), 
    # get year variable from image name
    year = trimws(sapply(strsplit(image_name, "_"), `[[`, 2)),
    year = sub("^.*_\\s+(.*?)\\s+_.*$", "\\1", year)
  ) %>%
  # keep only variables needed
  select(
    city, year, image_name, parcel_id, address, lat, lon, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm)
  # rename variables to standardize across datasets - N/A

# detroit 
# merge stage1 and stage2 preprocessing data 
# upload preprocessing results file
det_preprocessing <- 
  read_csv(file = "G:/My Drive/Changing Cities Research Lab/PROJECT FOLDER  GSV/preprocessing/preprocessing-results/final_image_sets/detroit_trash/detroit_trash_time_series_preprocessed.csv")
dat_det_ts <- 
  left_join(
    dat_det_ts,
    det_preprocessing %>% select(image_name, stage1_good, stage2_good)
  )
rm(det_preprocessing)

dat_det_ts <- 
  dat_det_ts %>%
  mutate(
    # add city variable
    city = "detroit", 
    parcel_id = as.character(parcel_id), 
    # get year variable from image name
    year = trimws(sapply(strsplit(image_name, "_"), `[[`, 2)),
    year = sub("^.*_\\s+(.*?)\\s+_.*$", "\\1", year)
  ) %>%
  # keep only variables needed
  select(
    city, year, image_name, parcel_id, address, lat, lon, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm)
  # rename variables to standardize across datasets - n/a

# la 
# merge address data from original parcels file
dat_la_ts <- 
  left_join(
    dat_la_ts %>% 
      mutate(
        # create parcel id variable from image name
        parcel_id = as.numeric(trimws(sapply(strsplit(image, "_"), `[[`, 1)))), 
    la_parcels %>% select(AIN, PropertyLocation), 
    by = c("parcel_id" = "AIN")
  )

dat_la_ts <- 
  dat_la_ts %>%
  mutate(
    # add city variable
    city = "la", 
    parcel_id = as.character(parcel_id), 
    # get year variable from image name
    year = trimws(sapply(strsplit(image, "_"), `[[`, 2)),
    year = sub("^.*_\\s+(.*?)\\s+_.*$", "\\1", year), 
    # create lat and lon variables
    lat = trimws(sapply(strsplit(lat_long, ",[ ]"), `[[`, 1)),
    lat = sub("[(]", "", lat),
    # create lat and lon variables
    lon = trimws(sapply(strsplit(lat_long, ",[ ]"), `[[`, 2)),
    lon = sub("[)]", "", lon),
    # create block group variable
    blkgrp = as.numeric(substr(block, 1, 1))
  ) %>%
  # rename variables to standard names 
  rename(
    address = PropertyLocation, 
    stage1_good = good_quality, 
    pred_svm = ml_prediction, 
    stage2_good = stage_2_good
  ) %>% 
  # keep only variables needed
  select(
    city, year, image, parcel_id, address, lat, lon, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm) %>%
  # rename variables to standardize across datasets
  rename(
    image_name = image
  )

# merge la geocodes with la data
# change parcel_id in la geo for merge
la_geo <- la_geo %>% mutate(parcel_id = as.character(parcel_id))
dat_la_ts <- 
  left_join(
    dat_la_ts, 
    la_geo
  ) %>% 
  mutate(
    tract = ifelse(is.na(tract), tract_add, tract), 
    blkgrp = ifelse(is.na(blkgrp), blkgrp_add, blkgrp), 
    block = ifelse(is.na(block), block_add, block)
  ) %>% 
  select(-tract_add, blkgrp_add, block_add)
rm(la_parcels, la_geo)

# philly
dat_phl_ts <- 
  dat_phl_ts %>%
  mutate(
    # add city variable
    city = "philly", 
    parcel_id = as.character(parcel_id), 
    lat = as.character(lat), 
    lon = as.character(lon), 
    year = as.character(year)
  ) %>%
  # keep only variables needed
  select(
    city, year, image, parcel_id, address, lat, lon, 
    tract, blkgrp, block, 
    stage1_good, stage2_good, pred_svm) %>%
  # rename variables to standardize across datasets
  rename(
    image_name = image
  )

# ID missing parcels that need geocoding ####
dat_aus_parcels_missing <- dat_aus_recent %>% filter(is.na(tract))
dat_la_parcels_missing <- dat_la_recent %>% filter(is.na(tract))

# combine datasets into single data frame ####
dat_recent <- 
  bind_rows(dat_aus_recent, dat_bos_recent, dat_det_recent, dat_la_recent, dat_phl_recent)

dat_timeseries <- 
  bind_rows(dat_aus_ts, dat_bos_ts, dat_det_ts, dat_la_ts, dat_phl_ts)


# create block face IDs ####
# using street names from "address"
dat_recent <- 
  dat_recent %>% 
  # clean address variable to standardize case and remove white space
  mutate(address = str_to_title(address), 
         address = str_squish(address)) %>% 
  # create street variable from address removing numbers and punctuation
  mutate(street =  str_remove_all(address, pattern = "\\,.*"), 
         street =  str_remove_all(street, pattern = "[0-9]+"),
         street =  str_replace_all(street, "[^a-zA-Z0-9]", " ")) %>% 
  # remove white space from street names
  mutate(street = str_squish(street)) %>% 
  # create a block face ID with block and street combo
  mutate(block_face = paste(block, street, sep= " - ")) %>% 
  # reorder variables
  select(city, image_name, parcel_id, address, street, lat, lon, 
         tract, blkgrp, block, block_face, 
         stage1_good, stage2_good, pred_svm)

dat_timeseries <- 
  dat_timeseries %>% 
  # clean address variable to standardize case and remove white space
  mutate(address = str_to_title(address), 
         address = str_squish(address)) %>% 
  # create street variable from address removing numbers and punctuation
  mutate(street =  str_remove_all(address, pattern = "\\,.*"), 
         street =  str_remove_all(street, pattern = "[0-9]+"),
         street =  str_replace_all(street, "[^a-zA-Z0-9]", " ")) %>% 
  # remove white space from street names
  mutate(street = str_squish(street)) %>% 
  # create a block face ID with block and street combo
  mutate(block_face = paste(block, street, sep= " - ")) %>% 
  # reorder variables
  select(city, year, image_name, parcel_id, address, street, lat, lon, 
         tract, blkgrp, block, block_face, 
         stage1_good, stage2_good, pred_svm)

# ACS data #-------------------------------------------------------------------
# clean data for merging
censusacs <- 
  censusacs %>% 
  select(trtid10, pop7a, hu7a, ppov7a, pcol7a, pown7a, pvac7a,  
         pnhwht7a, phisp7a, pasian7a, pnhblk7a, pfb7a, 
         mhmval7a, mrent7a)


# Save Results ------------------------------------------------------------

# export files of parcels with missing geocodes ####
# save to GDrive in workdir
filename <- "aus_parcels_missing_geocode.csv"
write_csv(dat_aus_parcels_missing, file = paste0(homedir, workdir, filename), na = "")

filename <- "la_parcels_missing_geocode.csv"
write_csv(dat_la_parcels_missing, file = paste0(homedir, workdir, filename), na = "")

# export final public files ####
# files do not fit in Github, saved on GDrive for now
filename <- "crossval_results_pa.csv"
write_csv(dat_crossval_pa, file = paste0(homedir, savedir, filename), na = "")
filename <- "crossval_results_lh.csv"
write_csv(dat_crossval_lh, file = paste0(homedir, savedir, filename), na = "")
filename <- "crossval_results_multi.csv"
write_csv(dat_crossval_multi, file = paste0(homedir, savedir, filename), na = "")

filename <- "final_results_recent.csv"
write_csv(dat_recent, file = paste0(homedir, savedir, filename), na = "")
filename <- "final_results_timeseries.csv"
write_csv(dat_timeseries, file = paste0(homedir, savedir, filename), na = "")

filename <- "acs7a_tract.csv"
write_csv(censusacs, file = paste0(homedir, savedir, filename), na = "")

filename <- "training_data_trash.csv"
write_csv(dat_train, file = paste0(homedir, savedir, filename), na = "")

filename <- "reliability_data_pairs.csv"
write_csv(dat_reliability_pairs, file = paste0(homedir, savedir, filename), na = "")
filename <- "reliability_data_single_image.csv"
write_csv(dat_reliability_single_image, file = paste0(homedir, savedir, filename), na = "")



