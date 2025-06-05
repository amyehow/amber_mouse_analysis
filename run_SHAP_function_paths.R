#Paths/Labels to SHAP values for each classifier to run SHAP_features_function
#Amy Howard 6-5-2025

library(tidyverse)
library(dplyr)
library(data.table)
library(scales) 
library(shapviz)
library(ggplot2)
library(xgboost)

source('./SHAP_features_function.R')

run_shap_analysis(
  shap_path = "Active Nursing/SHAP_values_active_nursing.csv",
  raw_path = "Active Nursing/RAW_SHAP_feature_values_active_nursing.csv",
  behavior_label = "active_nursing")

run_shap_analysis(
  shap_path = "Nest Attendance/SHAP_values_nest_attendance.csv",
  raw_path = "Nest Attendance/RAW_SHAP_feature_values_nest_attendance.csv",
  behavior_label = "nest_attendance")

run_shap_analysis(
  shap_path = "Passive Nursing/SHAP_values_passive_nursing.csv",
  raw_path = "Passive Nursing/RAW_SHAP_feature_values_passive_nursing.csv",
  behavior_label = "passive_nursing")

run_shap_analysis(
  shap_path = "Licking Grooming/SHAP_values_licking_grooming.csv",
  raw_path = "Licking Grooming/RAW_SHAP_feature_values_licking_grooming.csv",
  behavior_label = "licking_grooming")

run_shap_analysis(
  shap_path = "Nest Building/SHAP_values_nest_building.csv",
  raw_path = "Nest Building/RAW_SHAP_feature_values_nest_building.csv",
  behavior_label = "nest_building")

run_shap_analysis(
  shap_path = "Dam Self Grooming/SHAP_values_sd_grooming.csv",
  raw_path = "Dam Self Grooming/RAW_SHAP_feature_values_sd_grooming.csv",
  behavior_label = "sd_grooming")

run_shap_analysis(
  shap_path = "Dam Eating/SHAP_values_dam_eating.csv",
  raw_path = "Dam Eating/RAW_SHAP_feature_values_dam_eating.csv",
  behavior_label = "dam_eating")

run_shap_analysis(
  shap_path = "Dam Drinking/SHAP_values_dam_drinking.csv",
  raw_path = "Dam Drinking/RAW_SHAP_feature_values_dam_drinking.csv",
  behavior_label = "dam_drinking")



