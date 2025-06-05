#SHAP values analysis function
  #Script to create top 6 feature importance beeswarm plot
  #and sum of SHAP score by feature category lollipop plot
  #Amy Howard 5-6-2025

#SHAP (SHapley Additive exPlanations) is a game theoretic approach 
  #to explain the output of any machine learning model
#SHAP values decompose -as fair as possible- predictions into additive feature contributions.
  #an explainability metric to help uncover the black box of model structures
    #The contributions are implicitly normalized, which makes them easier to interpret and compare. 
    #If a feature’s value does not have any impact on the prediction, 
    #then it will be assigned a 0 contribution. If two features’ values have the a symmetrical impact 
    #across all subsets, they will be assigned equal contributions, 
    #and the local contributions are additive across instances

#We use a beeswarm plot to represent the top six features with the 
  #largest absolute SHAP scores for the behavior classifier 
  #where the solid black line is the base rate for the behavior 
  #(probability of a given frame containing the behavior by chance) 
#Each individual point reflects the change in behavior probability 
  #relative to the base rate (SHAP score) for that feature for one frame 
#The color of the point reflects the z-score of the actual feature value for that frame

#############
#SHAP/FEATURE IMPORTANCE FUNCTION

library(tidyverse)
library(dplyr)
library(data.table)
library(scales) 
library(shapviz)
library(ggplot2)
library(xgboost)

run_shap_analysis <- function(shap_path, raw_path, behavior_label) {
   save_path_prefix = "graphs/"
   df <- read.csv(shap_path)[, -1]  
   rawdf <- read.csv(raw_path)[, -1]
  
   
  #FEATURE IMPORTANCE BEESWARM PLOT
   
  #Get rid of SHAP meta data columns that we don't need
  #setdiff used to find elements that are in the first object but not in the second
   exclude_cols <- c("Unnamed: 0", "Expected_value", "Sum", "Prediction_probability", 
                     behavior_label, "frame")
   shap_cols <- setdiff(names(df), exclude_cols)
   
  #Extract SHAP and raw feature matrices
   shap_mat <- as.matrix(df[, shap_cols])
   X_raw <- rawdf[, shap_cols]  
    #make sure the feature names and row order match
  
  #SHAP and feature values stored in a "shapviz" object
   shp <- shapviz(shap_mat, X = X_raw)
   
  #Beeswarm plot (all features)
   #sv_importance(): Importance plot (bar/beeswarm).
   sv_importance(shp, kind = "bee")
   ggsave(filename = paste0(save_path_prefix, "bees_", behavior_label, "_all_features.png"),
          width = 6, height = 4.5)
  
  #Beeswarm plot (top 6 features, colored by actual feature values)
   #made top 6 graphc-what AMBER did
   sv_importance(shp, kind = "bee", max_display = 6) + 
     labs(title = "Nest Attendance",x = "SHAP Score Change from Base Rate") + 
     theme_bw()
   ggsave(filename = paste0(save_path_prefix, "bees_", behavior_label, "_top6.png"),
                 width = 6, height = 4.5)
   
   
   #LOLLIPOP PLOT BY FEATURE CATEGORIES
   
   #Feature-to-category mapping using patterns
   #This code is creating a new vector called feature_to_category that assigns 
   #each location in the shap_cols to a category based on pattern matching. 
   #The idea is to group similar feature names into 
   #broader categories (like "Dam movement," "Pup probabilities," etc.)
   #grepl searches for matches in characters or sequences present in a given string
   
   feature_to_category <- sapply(shap_cols, function(x) {
     if (grepl("movement", x)) {"Dam movement"}
     else if (grepl("high_p_dam|avg_dam_bp_p|sum_probabilities", x)) {"Dam probabilities"}
     else if (grepl("high_p_pup|pup_avg_p|arm_p", x)) {"Pup probabilities"}
     else if (grepl("pups_convex_hull", x)) {"Pup area"}
     else if (grepl("angle", x)) {"Dam angle"}
     else if (grepl("side_p", x)) {"Dam distance"}
     else if (grepl("convex_hull", x)) {"Dam area"}
     else if (grepl("pup.*distance|head_pup|dam_pup", x)) {"Dam-pup distance"}
     else if (grepl("centroid|arm_y|side_y", x)) {"Dam location"}
     else {"Other"}})
   
   names(feature_to_category) <- shap_cols
   
   #long-format SHAP + category data frame
   shap_long <- df %>%
     select(all_of(shap_cols)) %>%
     pivot_longer(cols = everything(), names_to = "Feature", values_to = "SHAP") %>%
     mutate(Category = feature_to_category[Feature])
   
   #Summarize SHAP by the 9 categories
   category_summary <- shap_long %>% group_by(Category) %>% 
     summarise(MeanSHAP = mean(SHAP, na.rm = TRUE), MinSHAP = min(SHAP, na.rm = TRUE),
               MaxSHAP = max(SHAP, na.rm = TRUE), .groups = "drop") %>%
     filter(Category != "Other") %>% mutate(Category = fct_reorder(Category, MeanSHAP))
   
   #Manual colors
   category_colors <- c(
     "Dam movement" = "#F34D98",
     "Dam probabilities" = "#7EC80A",
     "Pup probabilities" = "#006F3F",
     "Pup area" = "#A3D9FF",
     "Dam angle" = "#9B7CFF",
     "Dam distance" = "#5D9EFF",
     "Dam area" = "#F7C20B",
     "Dam-pup distance" = "#F18B4E",
     "Dam location" = "#F25D5A")
   
   #Lollipop plot
   ggplot(category_summary, aes(x = MeanSHAP, y = Category)) +
     geom_errorbarh(aes(xmin = MinSHAP, xmax = MaxSHAP), height = 0.2) +
     geom_point(aes(x = MinSHAP, color = Category), size = 4) +
     geom_point(aes(x = MaxSHAP, color = Category), size = 4) +
     geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
     scale_color_manual(values = category_colors) +
     labs(title = "Sum of SHAP Score by Feature Category",
          subtitle = behavior_label,
          x = "Sum of Mean SHAP Scores", y = "Feature Category") +
     theme_bw(base_size = 14) + theme(legend.position = "none")
     ggsave(filename = paste0(save_path_prefix, "shap_lollipop_", behavior_label, ".png"),
            width = 6, height = 4.5)
   }



