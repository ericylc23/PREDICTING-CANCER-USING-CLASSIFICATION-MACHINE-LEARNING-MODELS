## Load necessary libraries
library(tidyverse)
library(dplyr)
library(caret)
library(e1071) # SVM
library(randomForest)
library(class) # KNN
library(pROC) # ROC and AUC calculation
library(naivebayes)
library(rpart)
library(ROSE)
library(ROCR)
library(MLmetrics)
library(naivebayes)
library(FSelector)
library(Rmpfr)
library(ranger)
library(ROCR)
library(VIM)
library(smotefamily)
# Read the data
data <- read.csv("data-project.csv", header = TRUE)
#seed
set.seed(1000)
# Columns to delete, these columns contains no useful info
columns_to_delete <- c('FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE', 'SEQNO','QSTVER')

# Delete specified columns
data <- data[, !(names(data) %in% columns_to_delete)]

# List of columns with special codes 'not sure' & 'refuse to answer'
columns_with_special_codes <- c('GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'HLTHPLN1', 'PERSDOC2', 
                                'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'CHOLCHK2', 'TOLDHI2', 
                                'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3', 'CHCCOPD2', 
                                'ADDEPEV3', 'CHCKDNY2', 'DIABETE4', 'HAVARTH4', 'MARITAL', 
                                'EDUCA', 'RENTHOM1', 'CPDEMO1B', 'VETERAN3', 'EMPLOY1', 
                                'INCOME2', 'DEAF', 'BLIND', 'DECIDE', 'DIFFWALK', 
                                'DIFFDRES', 'DIFFALON', 'SMOKE100', 'USENOW3', 'EXERANY2', 
                                'STRENGTH', 'FLUSHOT7', 'TETANUS1', 'PNEUVAC4', 
                                'HIVTST7', 'HIVRISK5')

# Check which columns exist in the dataset
existing_columns <- columns_with_special_codes[columns_with_special_codes %in% colnames(data)]

# Replace special codes with NA
for (column in existing_columns) {
  data[[column]] <- replace(data[[column]], data[[column]] %in% c(7, 9, 77, 99, 999, 9999, 777, 7777), NA)
}

# Ensure all columns in dataset are numeric
data$SEXVAR <- as.numeric(data$SEXVAR)
###################################################################################################################################################
# Identify columns with missing values
missing_values <- colSums(is.na(data))
columns_with_missing_values <- names(missing_values[missing_values > 0])


# Function to check the number of NAs in a dataset
count_nas <- function(dataset) {
  na_count <- sapply(dataset, function(x) sum(is.na(x)))
  na_count_df <- data.frame(Column = names(na_count), NA_Count = na_count)
  return(na_count_df)
}

# Example usage
na_summary <- count_nas(data)
print(na_summary)

# List of columns with 'none' coded as '88', '8', '888', or '555'
columns_with_special_codes_as_none <- c('PHYSHLTH', 'MENTHLTH', 'FRUIT2', 'FRUITJU2', 
                                        'FVGREEN1', 'FRENCHF1', 'POTATOE1', 'VEGETAB2', 'CHILDREN')
columns_with_8_as_never <- c('CHECKUP1')
columns_with_888_as_none <- c('ALCDAY5')

# Replace '88' and '555' with '0'
for (column in columns_with_special_codes_as_none) {
  data[[column]] <- replace(data[[column]], data[[column]] %in% c(88, 555), 0)
}

# Replace '8' with '0'
for (column in columns_with_8_as_never) {
  data[[column]] <- replace(data[[column]], data[[column]] == 8, 0)
}

# Replace '888' with '0'
for (column in columns_with_888_as_none) {
  data[[column]] <- replace(data[[column]], data[[column]] == 888, 0)
}

# Classify the columns with missing values as numeric, categorical, or ordinal
numeric_columns <- intersect(columns_with_missing_values, c("PHYSHLTH", "MENTHLTH", "WEIGHT2", "HEIGHT3", "ALCDAY5", "FRUIT2", "FRUITJU2", 
                                                            "FVGREEN1", "FRENCHF1", "POTATOE1", "VEGETAB2", "HTIN4", "HTM4", "WTKG3", 
                                                            "CHILDREN"))
categorical_columns <- intersect(columns_with_missing_values, c("HLTHPLN1", "PERSDOC2", "MEDCOST", "BPHIGH4", 
                                                                "TOLDHI2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3", "CHCCOPD2", 
                                                                "ADDEPEV3", "CHCKDNY2", "DIABETE4", "HAVARTH4", "MARITAL", 
                                                                "RENTHOM1", "CPDEMO1B", "VETERAN3", "EMPLOY1", "DEAF", "BLIND", 
                                                                "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100", 
                                                                "USENOW3", "EXERANY2", "STRENGTH", "FLUSHOT7", "PNEUVAC4", 
                                                                "HIVTST7", "HIVRISK5", "QSTLANG", "TETANUS1", "DRNKANY5"))
ordinal_columns <- intersect(columns_with_missing_values, c("GENHLTH", "CHECKUP1", "EDUCA", "INCOME2", "CHOLCHK2"))

# Make a copy of the original data to compare later
original_data <- data

# Apply KNN imputation
imputed_data <- kNN(data[, numeric_columns], k = 8)

# Update the original data with imputed values
data[, numeric_columns] <- imputed_data[, numeric_columns]

# List of columns that should only have integer values
integer_columns <- c("CHILDREN", "PHYSHLTH", "MENTHLTH", "FRUIT2", "FRUITJU2", "FVGREEN1", "FRENCHF1", "POTATOE1", "VEGETAB2")

# Round the imputed values for these columns to the nearest integer
for (col in integer_columns) {
  data[[col]] <- round(data[[col]])
}

# Impute categorical columns using mode
impute_mode <- function(column) {
  mode_value <- as.character(names(sort(table(column), decreasing = TRUE))[1])
  return(replace(column, is.na(column), mode_value))
}
if (length(categorical_columns) > 0) {
  data[categorical_columns] <- lapply(data[categorical_columns], impute_mode)
}

# Impute ordinal columns using median
impute_median <- function(column) {
  median_value <- median(as.numeric(as.character(column)), na.rm = TRUE)
  return(replace(column, is.na(column), median_value))
}
if (length(ordinal_columns) > 0) {
  data[ordinal_columns] <- lapply(data[ordinal_columns], impute_median)
}

# Verify that there are no more missing values
missing_values_post_imputation <- colSums(is.na(data))
print(missing_values_post_imputation)
####################################################################################################################################################
# Standardize WEIGHT2 (convert kilograms to pounds)
data <- data %>%
  mutate(WEIGHT2 = ifelse(WEIGHT2 >= 9000, (WEIGHT2 - 9000) * 2.20462, WEIGHT2))

# Function to convert height in feet and inches to inches
convert_to_inches <- function(height) {
  feet <- floor(height / 100)
  inches <- height %% 100
  total_inches <- (feet * 12) + inches
  return(total_inches)
}

# Apply the conversion function to HEIGHT3, convert everything into inches
data <- data %>%
  mutate(HEIGHT3 = ifelse(HEIGHT3 >= 9000, (HEIGHT3 - 9000) / 2.54, convert_to_inches(HEIGHT3)))

# Check if columns exist before removing them
columns_to_remove <- c("HTIN4", "HTM4", "WTKG3")
existing_columns_to_remove <- intersect(columns_to_remove, colnames(data))

if (length(existing_columns_to_remove) > 0) {
  data <- data[, !(names(data) %in% existing_columns_to_remove)]
}

# Print the column names to verify the changes
print(colnames(data))

# UPDATE the new numeric columns
numeric_columns <- c("PHYSHLTH", "MENTHLTH", "WEIGHT2", "HEIGHT3", "ALCDAY5", "FRUIT2", 
                     "FRUITJU2", "FVGREEN1", "FRENCHF1", "POTATOE1", "VEGETAB2", "CHILDREN")

# Cap outliers
cap_outliers <- function(column) {
  Q1 <- quantile(column, 0.25, na.rm = TRUE)
  Q3 <- quantile(column, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  column <- ifelse(column < lower_bound, lower_bound, column)
  column <- ifelse(column > upper_bound, upper_bound, column)
  
  return(column)
}

for (col in numeric_columns) {
  data[[col]] <- cap_outliers(data[[col]])
}

# Identify categorical and ordinal columns
categorical_columns <- c("HLTHPLN1", "PERSDOC2", "MEDCOST", "BPHIGH4", 
                         "TOLDHI2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3", 
                         "CHCCOPD2", "ADDEPEV3", "CHCKDNY2", "DIABETE4", "HAVARTH4", "MARITAL", "RENTHOM1", "CPDEMO1B", "VETERAN3", "EMPLOY1", 
                         "DEAF", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", 
                         "SMOKE100", "USENOW3", "EXERANY2", "STRENGTH", "FLUSHOT7", 
                         "PNEUVAC4", "HIVTST7", "HIVRISK5", "QSTLANG", "TETANUS1", "DRNKANY5")

ordinal_columns <- c("GENHLTH", "CHECKUP1", "EDUCA", "INCOME2","CHOLCHK2")

# Convert categorical columns to factors
data[categorical_columns] <- lapply(data[categorical_columns], as.factor)

# Create dummy variables
dummies <- dummyVars(~ ., data = data[categorical_columns])

# Generate the one-hot encoded data
one_hot_encoded <- predict(dummies, newdata = data[categorical_columns])

# Combine the one-hot encoded columns with the original dataset
data <- cbind(data, one_hot_encoded)

# Remove the original categorical columns
data <- data[, !(names(data) %in% categorical_columns)]

# Check the results
summary(data)

# Identify the numeric columns for scaling
numeric_columns <- c("PHYSHLTH", "MENTHLTH", "WEIGHT2", "HEIGHT3", "ALCDAY5", "FRUIT2", "FRUITJU2", 
                     "FVGREEN1", "FRENCHF1", "POTATOE1", "VEGETAB2", "CHILDREN")

# Apply min-max scaling
preprocessParams <- preProcess(data[numeric_columns], method = c("range"))
data[numeric_columns] <- predict(preprocessParams, data[numeric_columns])

# Check the results
summary(data[numeric_columns])

# Encode the Class column: 'Y' to 1 and 'N' to 0
data$Class <- ifelse(data$Class == 'Y', 1, 0)

# Verify the encoding
table(data$Class)


# Set the seed 
set.seed(1023)

# Determine the split ratio (80% training, 34% testing)
split_ratio <- 0.8

# Create the train-test split using stratified splitting
train_index <- createDataPartition(data$Class, p = split_ratio, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

print(table(train_data$Class))

# Verify the split
cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")
print(table(train_data$Class))
print(table(test_data$Class))
# Save to CSV files
write.csv(data, "processed_data.csv", row.names = FALSE)
write.csv(train_data, "initial_train.csv", row.names = FALSE)
write.csv(test_data, "initial_test.csv", row.names = FALSE)
print("Train and test datasets have been saved successfully.")

###############################################
#data validation 
# Check for overlapping rows
overlapping_rows <- intersect(rownames(train_data), rownames(test_data))
cat("Number of overlapping rows:", length(overlapping_rows), "\n")
if (length(overlapping_rows) > 0) {
  cat("Overlapping row indices:", overlapping_rows, "\n")
} else {
  cat("No overlapping rows found.\n")
}

# Check for missing values in training and testing sets
missing_values_train <- colSums(is.na(train_data))
missing_values_test <- colSums(is.na(test_data))

cat("Missing values in training data:\n")
print(missing_values_train[missing_values_train > 0])

cat("Missing values in testing data:\n")
print(missing_values_test[missing_values_test > 0])

# Verify if there are any missing values
if (all(missing_values_train == 0)) {
  cat("No missing values in training data.\n")
} else {
  cat("There are missing values in the training data.\n")
}

if (all(missing_values_test == 0)) {
  cat("No missing values in testing data.\n")
} else {
  cat("There are missing values in the testing data.\n")
}

# Check data types in the training and testing sets
data_types_train <- sapply(train_data, class)
data_types_test <- sapply(test_data, class)

cat("Data types in training data:\n")
print(data_types_train)

cat("Data types in testing data:\n")
print(data_types_test)

# Check class distribution in training and testing sets
class_distribution_train <- table(train_data$Class)
class_distribution_test <- table(test_data$Class)

cat("Class distribution in training data:\n")
print(class_distribution_train)

cat("Class distribution in testing data:\n")
print(class_distribution_test)

# Identify duplicates in the training and testing sets
train_duplicates <- train_data[duplicated(train_data), ]
test_duplicates <- test_data[duplicated(test_data), ]

cat("Number of duplicates in training data:", nrow(train_duplicates), "\n")
cat("Number of duplicates in testing data:", nrow(test_duplicates), "\n")
# Ensure Class is a factor
data$Class <- factor(data$Class)
# Verify the change
str(data$Class)

# Ensure Class column is a factor in training and test data
train_data$Class <- as.factor(train_data$Class)
test_data$Class <- as.factor(test_data$Class)

# Adjust the N parameter for ROS
minority_class_count <- sum(train_data$Class == "1")
majority_class_count <- sum(train_data$Class == "0")
desired_count <- max(minority_class_count, majority_class_count)

ros_train_data <- ovun.sample(Class ~ ., data = train_data, method = "over", N = desired_count * 2)$data
ros_train_data <- as.data.frame(ros_train_data)

# Verify the new Class distribution
print(table(ros_train_data$Class))

# Post-Balancing Data Validation for ROS
missing_values_ros <- colSums(is.na(ros_train_data))
cat("Missing values in ROS training data:\n")
print(missing_values_ros[missing_values_ros > 0])

ros_duplicates <- ros_train_data[duplicated(ros_train_data), ]
cat("Number of duplicates in ROS training data:", nrow(ros_duplicates), "\n")

data_types_ros <- sapply(ros_train_data, class)
cat("Data types in ROS training data:\n")
print(data_types_ros)

cat("Class distribution in ROS training data after balancing:\n")
print(table(ros_train_data$Class))

# Apply RUS
rus_train_data <- ovun.sample(Class ~ ., data = train_data, method = "under", N = 2 * sum(train_data$Class == "1"))$data
rus_train_data <- as.data.frame(rus_train_data)

# Verify the Class column and distribution in RUS data
print(colnames(rus_train_data))
print(table(rus_train_data$Class))

# Post-Balancing Data Validation for RUS
missing_values_rus <- colSums(is.na(rus_train_data))
cat("Missing values in RUS training data:\n")
print(missing_values_rus[missing_values_rus > 0])

rus_duplicates <- rus_train_data[duplicated(rus_train_data), ]
cat("Number of duplicates in RUS training data:", nrow(rus_duplicates), "\n")

data_types_rus <- sapply(rus_train_data, class)
cat("Data types in RUS training data:\n")
print(data_types_rus)

cat("Class distribution in RUS training data after balancing:\n")
print(table(rus_train_data$Class))

write.csv(train_data, "rus_train_data.csv", row.names = FALSE)
write.csv(test_data, "ros_train_data.csv", row.names = FALSE)

# Original class distribution
cat("Original class distribution in training data:\n")
print(table(train_data$Class))

# Ensure the Class column is a factor
train_data$Class <- as.numeric(as.factor(train_data$Class)) - 1

# Apply SMOTE to the training data
smote_train_data <- SMOTE(X = train_data[, -ncol(train_data)], target = train_data$Class, K = 5)

# Extract the new data and ensure it's a data frame
balanced_train_data <- smote_train_data$data
balanced_train_data <- as.data.frame(balanced_train_data)

# Convert Class column back to factor
balanced_train_data$Class <- factor(balanced_train_data$class, levels = c(0, 1))
balanced_train_data$class <- NULL  # Remove the extra class column created by SMOTE

# Verify the new Class distribution
cat("Class distribution in SMOTE training data:\n")
print(table(balanced_train_data$Class))

# Ensure the Class column is a factor
train_data$Class <- as.factor(train_data$Class)

# Save the processed data
write.csv(balanced_train_data, "smote_train_data.csv", row.names = FALSE)
######################################################
# Attribute selection

# Register parallel backend
library(doParallel)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
# Define control for RFE
r_control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)

# RFE on ROS Data
set.seed(1023)
rfe_model_ros <- caret::rfe(Class ~ ., data = ros_train_data, sizes = c(1:10, 15, 20, 25), rfeControl = r_control)
rfe_selected_features_ros <- predictors(rfe_model_ros)
print("RFE Selected Features on ROS Data:")
print(rfe_selected_features_ros)

# RFE on RUS Data
set.seed(1023)
rfe_model_rus <- caret::rfe(Class ~ ., data = rus_train_data, sizes = c(1:10, 15, 20, 25), rfeControl = r_control)
rfe_selected_features_rus <- predictors(rfe_model_rus)
print("RFE Selected Features on RUS Data:")
print(rfe_selected_features_rus)

# RFE on SMOTE Data
set.seed(1023)
rfe_model_smote <- caret::rfe(Class ~ ., data = balanced_train_data, sizes = c(1:10, 15, 20, 25), rfeControl = r_control)
rfe_selected_features_smote <- predictors(rfe_model_smote)
print("RFE Selected Features on SMOTE Data:")
print(rfe_selected_features_smote)

# Apply Random Forest Importance on ROS balanced data
set.seed(1023)
rf_model_ros <- randomForest(Class ~ ., data = ros_train_data, importance = TRUE)
rf_importance_ros <- randomForest::importance(rf_model_ros)
rf_selected_features_ros <- rownames(rf_importance_ros)[order(rf_importance_ros[, "MeanDecreaseGini"], decreasing = TRUE)[1:25]]
print("Random Forest Selected Features for ROS data:")
print(rf_selected_features_ros)

# Apply Random Forest Importance on RUS balanced data
set.seed(1023)
rf_model_rus <- randomForest(Class ~ ., data = rus_train_data, importance = TRUE)
rf_importance_rus <- randomForest::importance(rf_model_rus)
rf_selected_features_rus <- rownames(rf_importance_rus)[order(rf_importance_rus[, "MeanDecreaseGini"], decreasing = TRUE)[1:25]]
print("Random Forest Selected Features for RUS data:")
print(rf_selected_features_rus)

# Apply Random Forest Importance on SMOTE balanced data
set.seed(1023)
rf_model_smote <- randomForest(Class ~ ., data = balanced_train_data, importance = TRUE)
rf_importance_smote <- randomForest::importance(rf_model_smote)
rf_selected_features_smote <- rownames(rf_importance_smote)[order(rf_importance_smote[, "MeanDecreaseGini"], decreasing = TRUE)[1:25]]
print("Random Forest Selected Features for SMOTE data:")
print(rf_selected_features_smote)

# Apply Information Gain on ROS balanced data
# Ensure the Class column is a factor
#ros_train_data$Class <- as.factor(ros_train_data$Class)
#rus_train_data$Class <- as.factor(rus_train_data$Class)

ig_results_ros <- information.gain(Class ~ ., data = ros_train_data)
# Ensure that the results are sorted correctly and the top 20 features are extracted 
ig_results_ros_sorted <- ig_results_ros[order(ig_results_ros$attr_importance, decreasing = TRUE), , drop = FALSE]

# Extract the top 20 features based on Information Gain
set.seed(1023)
top_n_features_ig_ros <- rownames(ig_results_ros_sorted)[1:20]

print("Top 20 features based on Information Gain for ROS data:")
print(top_n_features_ig_ros)

# Apply Information Gain on RUS balanced data
ig_results_rus <- information.gain(Class ~ ., data = rus_train_data)

# Ensure that the results are sorted correctly and the top 20 features are extracted properly
ig_results_rus_sorted <- ig_results_rus[order(ig_results_rus$attr_importance, decreasing = TRUE), , drop = FALSE]
set.seed(1023)
# Extract the top 20 features based on Information Gain
top_n_features_ig_rus <- rownames(ig_results_rus_sorted)[1:20]

print("Top 20 features based on Information Gain for RUS data:")
print(top_n_features_ig_rus)

# Apply Information Gain on smote balanced data
ig_results_smote <- information.gain(Class ~ ., data = balanced_train_data)

# Ensure that the results are sorted correctly and the top 20 features are extracted properly
ig_results_smote_sorted <- ig_results_smote[order(ig_results_smote$attr_importance, decreasing = TRUE), , drop = FALSE]
set.seed(1023)
# Extract the top 20 features based on Information Gain
top_n_features_ig_smote <- rownames(ig_results_smote_sorted)[1:20]

print("Top 20 features based on Information Gain for smote data:")
print(top_n_features_ig_smote)

# Load necessary libraries for Boruta feature selection method
library(Boruta)

# Function to apply Boruta and return selected features
apply_boruta <- function(data) {
  # Ensure the Class column is a factor
  data$Class <- as.factor(data$Class)
  
  # Apply Boruta
  boruta_result <- Boruta(Class ~ ., data = data, doTrace = 0)
  
  # Extract confirmed attributes
  confirmed_attributes <- getSelectedAttributes(boruta_result, withTentative = FALSE)
  
  return(confirmed_attributes)
}

# Apply Boruta on ROS balanced data
set.seed(1023)
selected_features_ros <- apply_boruta(ros_train_data)
print("Boruta Selected Features for ROS data:")
print(selected_features_ros)
if (length(selected_features_ros) > 0) {
  ros_boruta_features <- ros_train_data[, c(selected_features_ros, "Class")]
} else {
  cat("No features selected for ROS data using Boruta.\n")
}

# Apply Boruta on RUS balanced data
set.seed(1023)
selected_features_rus <- apply_boruta(rus_train_data)
print("Boruta Selected Features for RUS data:")
print(selected_features_rus)
if (length(selected_features_rus) > 0) {
  rus_boruta_features <- rus_train_data[, c(selected_features_rus, "Class")]
} else {
  cat("No features selected for RUS data using Boruta.\n")
}

# Apply Boruta on SMOTE balanced data
set.seed(1023)
selected_features_smote <- apply_boruta(balanced_train_data)
print("Boruta Selected Features for smote data:")
print(selected_features_smote)
if (length(selected_features_smote) > 0) {
  smote_boruta_features <- balanced_train_data[, c(selected_features_smote, "Class")]
} else {
  cat("No features selected for smote data using Boruta.\n")
}

# Print structure of the feature-selected datasets to verify
if (exists("ros_boruta_features")) str(ros_boruta_features)
if (exists("rus_boruta_features")) str(rus_boruta_features)
if (exists("smote_boruta_features")) str(smote_boruta_features)

# Load necessary library for Chi-Square test
library(FSelector)
# Apply Chi-Square Test on ROS balanced data
set.seed(1023)
chi_sq_ros <- chi.squared(Class ~ ., data = ros_train_data)
chi_sq_sorted_ros <- chi_sq_ros[order(chi_sq_ros$attr_importance, decreasing = TRUE), , drop = FALSE]
selected_features_chi_sq_ros <- rownames(chi_sq_sorted_ros)[1:50]  # Select top 50 features

# Apply Chi-Square Test on RUS balanced data
set.seed(1023)
chi_sq_rus <- chi.squared(Class ~ ., data = rus_train_data)
chi_sq_sorted_rus <- chi_sq_rus[order(chi_sq_rus$attr_importance, decreasing = TRUE), , drop = FALSE]
selected_features_chi_sq_rus <- rownames(chi_sq_sorted_rus)[1:50]  # Select top 50 features

# Apply Chi-Square Test on SMOTE balanced data
set.seed(1023)
chi_sq_smote <- chi.squared(Class ~ ., data = balanced_train_data)
chi_sq_sorted_smote <- chi_sq_smote[order(chi_sq_smote$attr_importance, decreasing = TRUE), , drop = FALSE]
selected_features_chi_sq_smote <- rownames(chi_sq_sorted_smote)[1:50]  # Select top 50 features

# Create datasets with selected features for ROS
ros_chi_sq_features <- ros_train_data[, c(selected_features_chi_sq_ros, "Class")]

# Create datasets with selected features for RUS
rus_chi_sq_features <- rus_train_data[, c(selected_features_chi_sq_rus, "Class")]

# Create datasets with selected features for SMOTE
smote_chi_sq_features <- balanced_train_data[, c(selected_features_chi_sq_smote, "Class")]

# Print the structure of the feature-selected datasets to verify
str(ros_chi_sq_features)
str(rus_chi_sq_features)
str(smote_chi_sq_features)

set.seed(1023)
# Create datasets with selected features
# RFE Selected Features on ROS
ros_rfe_features <- ros_train_data[, c(rfe_selected_features_ros, "Class")]

# RFE Selected Features on smote
smote_rfe_features <- balanced_train_data[, c(rfe_selected_features_smote, "Class")]

# Random Forest Selected Features on ROS
ros_rf_features <- ros_train_data[, c(rf_selected_features_ros, "Class")]

# Random Forest Selected Features on SMOTE
smote_rf_features <- balanced_train_data[, c(rf_selected_features_smote, "Class")]

# Information Gain Selected Features on ROS
ros_ig_features <- ros_train_data[, c(top_n_features_ig_ros, "Class")]

# Information Gain Selected Features on smote
smote_ig_features <- balanced_train_data[, c(top_n_features_ig_smote, "Class")]

# Boruta Selected Features on ROS
ros_boruta_features 

# Boruta Selected Features on SMOTE
smote_boruta_features 

#chi-sqaured selected Features on ROS
ros_chi_sq_features

#chi-sqaured selected Features on SMOTE
smote_chi_sq_features

# RFE Selected Features on RUS
rus_rfe_features <- rus_train_data[, c(rfe_selected_features_rus, "Class")]

# RFE Selected Features on smote
smote_rfe_features <- balanced_train_data[, c(rfe_selected_features_smote, "Class")]

# Random Forest Selected Features on RUS
rus_rf_features <- rus_train_data[, c(rf_selected_features_rus, "Class")]

# Random Forest Selected Features on RUS
smote_rf_features <- balanced_train_data[, c(rf_selected_features_smote, "Class")]

# Information Gain Selected Features on RUS
rus_ig_features <- rus_train_data[, c(top_n_features_ig_rus, "Class")]

# Information Gain Selected Features on RUS
smote_ig_features <- balanced_train_data[, c(top_n_features_ig_smote, "Class")]

# Boruta Selected Features on RUS
rus_boruta_features 

#chi-sqaured selected Features on RUS
rus_chi_sq_features

# Print structure of the feature-selected datasets
str(ros_rfe_features)
str(ros_rf_features)
str(ros_ig_features)
str(ros_boruta_features)
str(rus_rfe_features)
str(rus_rf_features)
str(rus_ig_features)
str(rus_boruta_features)
str(rus_chi_sq_features)
str(ros_chi_sq_features)
str(smote_rfe_features)
str(smote_rf_features)
str(smote_ig_features)
str(smote_boruta_features)
str(smote_chi_sq_features)


# Function to rename class levels to rename the Class levels into "Class0" and "Class1"
rename_class_levels <- function(data) {
  levels(data$Class) <- c("Class0", "Class1")
  return(data)
}


ros_rfe_features <- rename_class_levels(ros_rfe_features)
ros_rf_features <- rename_class_levels(ros_rf_features)
ros_ig_features <- rename_class_levels(ros_ig_features)
ros_boruta_features <- rename_class_levels(ros_boruta_features)
ros_chi_sq_features <- rename_class_levels(ros_chi_sq_features)
rus_rfe_features <- rename_class_levels(rus_rfe_features)
rus_rf_features <- rename_class_levels(rus_rf_features)
rus_ig_features <- rename_class_levels(rus_ig_features)
rus_boruta_features <- rename_class_levels(rus_boruta_features)
rus_chi_sq_features <- rename_class_levels(rus_chi_sq_features)
smote_rfe_features <- rename_class_levels(smote_rfe_features)
smote_rf_features <- rename_class_levels(smote_rf_features)
smote_ig_features <- rename_class_levels(smote_ig_features)
smote_boruta_features <- rename_class_levels(smote_boruta_features)
smote_chi_sq_features <- rename_class_levels(smote_chi_sq_features)


feature_datasets <- list(
  ros_rfe = ros_rfe_features,
  ros_rf = ros_rf_features,
  ros_boruta = ros_boruta_features,
  ros_ig = ros_ig_features,
  ros_chi_sq = ros_chi_sq_features,
  rus_rfe = rus_rfe_features,
  rus_rf = rus_rf_features,
  rus_ig = rus_ig_features,
  rus_boruta = rus_boruta_features,
  rus_chi_sq = rus_chi_sq_features,
  smote_rfe = smote_rfe_features,
  smote_rf = smote_rf_features,
  smote_boruta = smote_boruta_features,
  smote_ig = smote_ig_features,
  smote_chi_sq = smote_chi_sq_features
  
)

# Ensure the Class column has exactly two levels in each dataset
for (dataset_name in names(feature_datasets)) {
  dataset <- feature_datasets[[dataset_name]]
  dataset$Class <- factor(dataset$Class, levels = c("Class0", "Class1"))
  feature_datasets[[dataset_name]] <- dataset
}

# Map the values to "Class0" and "Class1"
test_data$Class <- ifelse(test_data$Class == "0", "Class0",
                          ifelse(test_data$Class == "1", "Class1", NA))

# Ensure test_data Class column has correct levels and no NAs
test_data$Class <- factor(test_data$Class, levels = c("Class0", "Class1"))

# Function to apply selected features to test data
apply_selected_features_to_test <- function(test_data, selected_features) {
  # to ensure the test data has the same feature selection applied as the training data
  test_data_processed <- test_data  
  
  # Subset the test data to include only the selected features
  selected_features <- c(selected_features, "Class")
  test_data_selected <- test_data_processed[, colnames(test_data_processed) %in% selected_features]
  return(test_data_selected)
}

# Apply selected features to test data for each feature selection method
test_data_ros_rfe <- apply_selected_features_to_test(test_data, rfe_selected_features_ros)
test_data_rus_rfe <- apply_selected_features_to_test(test_data, rfe_selected_features_rus)
test_data_smote_rfe <- apply_selected_features_to_test(test_data, rfe_selected_features_smote)

test_data_ros_rf <- apply_selected_features_to_test(test_data, rf_selected_features_ros)
test_data_rus_rf <- apply_selected_features_to_test(test_data, rf_selected_features_rus)
test_data_smote_rf <- apply_selected_features_to_test(test_data, rf_selected_features_smote)

test_data_ros_ig <- apply_selected_features_to_test(test_data, top_n_features_ig_ros)
test_data_rus_ig <- apply_selected_features_to_test(test_data, top_n_features_ig_rus)
test_data_smote_ig <- apply_selected_features_to_test(test_data, top_n_features_ig_smote)

test_data_ros_boruta <- apply_selected_features_to_test(test_data, selected_features_ros)
test_data_rus_boruta <- apply_selected_features_to_test(test_data, selected_features_rus)
test_data_smote_boruta <- apply_selected_features_to_test(test_data, selected_features_smote)

test_data_ros_chi_sq <- apply_selected_features_to_test(test_data, selected_features_chi_sq_ros)
test_data_rus_chi_sq <- apply_selected_features_to_test(test_data, selected_features_chi_sq_rus)
test_data_smote_chi_sq <- apply_selected_features_to_test(test_data, selected_features_chi_sq_smote)


###############################################################################
#function setup for calculate performance metrix
# Function to calculate MCC
calculate_mcc <- function(TP, TN, FP, FN) {
  TP <- mpfr(TP, precBits = 256)
  TN <- mpfr(TN, precBits = 256)
  FP <- mpfr(FP, precBits = 256)
  FN <- mpfr(FN, precBits = 256)
  
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  if (denominator == 0) {
    mcc <- 0
  } else {
    mcc <- (TP * TN - FP * FN) / denominator
  }
  return(as.numeric(mcc))  # Convert back to numeric
}

# Function to calculate Kappa
calculate_kappa <- function(TP, TN, FP, FN) {
  total <- as.numeric(TP + FP + TN + FN)
  p_o <- as.numeric((TP + TN) / total)
  p_e1 <- as.numeric(((TP + FN) / total) * ((TP + FP) / total))
  p_e2 <- as.numeric(((FP + TN) / total) * ((FN + TN) / total))
  p_e <- p_e1 + p_e2
  kappa <- (p_o - p_e) / (1 - p_e)
  return(kappa)
}

# Function to calculate performance metrics
calculate_performance_metrics <- function(model, test_data) {
  # Predict the classes
  predictions <- predict(model, test_data)
  # Predict the probabilities
  probabilities <- predict(model, test_data, type = "prob")
  
  # Ensure actuals is a factor with the correct levels
  actuals <- factor(test_data$Class, levels = c("Class0", "Class1"))
  
  # Calculate the confusion matrix
  conf_matrix <- table(Predicted = predictions, Actual = actuals)
  
  # Compute metrics for Class0
  TP0 <- conf_matrix["Class0", "Class0"]
  TN0 <- conf_matrix["Class1", "Class1"]
  FP0 <- conf_matrix["Class0", "Class1"]
  FN0 <- conf_matrix["Class1", "Class0"]
  
  tpr0 <- TP0 / (TP0 + FN0)
  fpr0 <- FP0 / (FP0 + TN0)
  precision0 <- TP0 / (TP0 + FP0)
  recall0 <- tpr0
  f_measure0 <- 2 * ((precision0 * recall0) / (precision0 + recall0))
  mcc0 <- calculate_mcc(TP0, TN0, FP0, FN0)
  kappa0 <- calculate_kappa(TP0, TN0, FP0, FN0)
  
  # Compute metrics for Class1
  TP1 <- conf_matrix["Class1", "Class1"]
  TN1 <- conf_matrix["Class0", "Class0"]
  FP1 <- conf_matrix["Class1", "Class0"]
  FN1 <- conf_matrix["Class0", "Class1"]
  
  tpr1 <- TP1 / (TP1 + FN1)
  fpr1 <- FP1 / (FP1 + TN1)
  precision1 <- TP1 / (TP1 + FP1)
  recall1 <- tpr1
  f_measure1 <- 2 * ((precision1 * recall1) / (precision1 + recall1))
  mcc1 <- calculate_mcc(TP1, TN1, FP1, FN1)
  kappa1 <- calculate_kappa(TP1, TN1, FP1, FN1)
  
  # Calculate ROC and AUC
  roc_predictions <- prediction(probabilities[, "Class1"], as.numeric(actuals) - 1)
  roc_performance <- performance(roc_predictions, measure = "tpr", x.measure = "fpr")
  auc_value <- performance(roc_predictions, measure = "auc")@y.values[[1]]
  
  # Weighted Average
  total <- sum(conf_matrix)
  weights <- c(sum(actuals == "Class0") / total, sum(actuals == "Class1") / total)
  tpr_avg <- sum(tpr0 * weights[1], tpr1 * weights[2])
  fpr_avg <- sum(fpr0 * weights[1], fpr1 * weights[2])
  precision_avg <- sum(precision0 * weights[1], precision1 * weights[2])
  recall_avg <- sum(recall0 * weights[1], recall1 * weights[2])
  f_measure_avg <- sum(f_measure0 * weights[1], f_measure1 * weights[2])
  mcc_avg <- sum(mcc0 * weights[1], mcc1 * weights[2])
  kappa_avg <- sum(kappa0 * weights[1], kappa1 * weights[2])
  
  # Create the performance measures table
  performance_table <- data.frame(
    TPR = c(tpr0, tpr1, tpr_avg),
    FPR = c(fpr0, fpr1, fpr_avg),
    Precision = c(precision0, precision1, precision_avg),
    Recall = c(recall0, recall1, recall_avg),
    F_measure = c(f_measure0, f_measure1, f_measure_avg),
    ROC = rep(auc_value, 3),
    MCC = c(mcc0, mcc1, mcc_avg),
    Kappa = c(kappa0, kappa1, kappa_avg),
    row.names = c("Class0", "Class1", "Wt. Average")
  )
  
  return(list(performance_table = performance_table, conf_matrix = conf_matrix, roc_performance = roc_performance))
}

###############################################################################
###############################################################################
########################################
#Classification models

# Suppress warnings
oldw <- getOption("warn")
options(warn = -1)

# Control for cross-validation
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)

# Initialize lists to store results
performance_tables_glm <- list()
confusion_matrices_glm <- list()

performance_tables_rpart <- list()
confusion_matrices_rpart <- list()

performance_tables_rf <- list()
confusion_matrices_rf <- list()

performance_tables_svm <- list()
confusion_matrices_svm <- list()

performance_tables_nb <- list()
confusion_matrices_nb <- list()

performance_tables_knn <- list()
confusion_matrices_knn <- list()

# Function to evaluate the model and save results
evaluate_and_save_results <- function(model, test_data, dataset_name, model_name) {
  # Evaluate the model
  evaluation <- calculate_performance_metrics(model, test_data)
  performance_table <- evaluation$performance_table
  conf_matrix <- evaluation$conf_matrix
  roc_performance <- evaluation$roc_performance
  
  # Print the confusion matrix
  cat("\nConfusion Matrix for", model_name, "model on", dataset_name, "dataset:\n")
  print(conf_matrix)
  
  # Print the performance table
  cat("\nPerformance Table for", model_name, "model on", dataset_name, "dataset:\n")
  print(performance_table)
  
  # Save the performance table to a CSV file
  write.csv(performance_table, file = paste0("performance_metrics_", model_name, "_", dataset_name, ".csv"), row.names = TRUE)
  
  # Save the confusion matrix to a CSV file
  write.csv(as.data.frame.matrix(conf_matrix), file = paste0("confusion_matrix_", model_name, "_", dataset_name, ".csv"), row.names = TRUE)
  
  # Save the ROC plot
  png(filename = paste0("roc_curve_", model_name, "_", dataset_name, ".png"))
  plot(roc_performance, main = paste("ROC Curve for", model_name, "on", dataset_name), col = "blue")
  abline(a = 0, b = 1, col = "gray", lty = 2)
  dev.off()
}



#Control for cross-validation
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)


#GLM model training
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
# Train and evaluate GLM model on ros_rfe dataset
cat("\nTraining GLM model on ros_rfe dataset...\n")
set.seed(1023)
glm_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "glm")

# Train and evaluate GLM model on ros_rf dataset
cat("\nTraining GLM model on ros_rf dataset...\n")
set.seed(1023)
glm_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_ros_rf, test_data_ros_rf, "ros_rf", "glm")

# Train and evaluate GLM model on ros_boruta dataset
cat("\nTraining GLM model on ros_boruta dataset...\n")
set.seed(1023)
glm_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "glm")

# Train and evaluate GLM model on ros_ig dataset
cat("\nTraining GLM model on ros_ig dataset...\n")
set.seed(1023)
glm_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_ros_ig, test_data_ros_ig, "ros_ig", "glm")

# Train and evaluate GLM model on ros_chi_sq dataset
cat("\nTraining GLM model on ros_chi_sq dataset...\n")
set.seed(1023)
glm_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "glm")

# Train and evaluate GLM model on rus_rfe dataset
cat("\nTraining GLM model on rus_rfe dataset...\n")
set.seed(1023)
glm_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "glm")

# Train and evaluate GLM model on rus_rf dataset
cat("\nTraining GLM model on rus_rf dataset...\n")
set.seed(1023)
glm_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_rus_rf, test_data_rus_rf, "rus_rf", "glm")

# Train and evaluate GLM model on rus_boruta dataset
cat("\nTraining GLM model on rus_boruta dataset...\n")
set.seed(1023)
glm_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "glm")

# Train and evaluate GLM model on rus_ig dataset
cat("\nTraining GLM model on rus_ig dataset...\n")
set.seed(1023)
glm_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_rus_ig, test_data_rus_ig, "rus_ig", "glm")

# Train and evaluate GLM model on rus_chi_sq dataset
cat("\nTraining GLM model on rus_chi_sq dataset...\n")
set.seed(1023)
glm_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "glm")

# Train and evaluate GLM model on smote_rfe dataset
cat("\nTraining GLM model on smote_rfe dataset...\n")
set.seed(1023)
glm_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "glm")

# Train and evaluate GLM model on smote_rf dataset
cat("\nTraining GLM model on smote_rf dataset...\n")
set.seed(1023)
glm_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_smote_rf, test_data_smote_rf, "smote_rf", "glm")

# Train and evaluate GLM model on smote_boruta dataset
cat("\nTraining GLM model on smote_boruta dataset...\n")
set.seed(1023)
glm_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "glm")

# Train and evaluate GLM model on smote_ig dataset
cat("\nTraining GLM model on smote_ig dataset...\n")
set.seed(1023)
glm_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_smote_ig, test_data_smote_ig, "smote_ig", "glm")

# Train and evaluate GLM model on smote_chi_sq dataset
cat("\nTraining GLM model on smote_chi_sq dataset...\n")
set.seed(1023)
glm_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "glm", trControl = control, metric = "ROC")
evaluate_and_save_results(glm_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "glm")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

###############################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# RPART Models
cat("\nTraining RPART model on ros_rfe dataset...\n")
set.seed(1023)
rpart_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "rpart")

cat("\nTraining RPART model on ros_rf dataset...\n")
set.seed(1023)
rpart_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_ros_rf, test_data_ros_rf, "ros_rf", "rpart")

cat("\nTraining RPART model on ros_boruta dataset...\n")
set.seed(1023)
rpart_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "rpart")

cat("\nTraining RPART model on ros_ig dataset...\n")
set.seed(1023)
rpart_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_ros_ig, test_data_ros_ig, "ros_ig", "rpart")

cat("\nTraining RPART model on ros_chi_sq dataset...\n")
set.seed(1023)
rpart_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "rpart")

cat("\nTraining RPART model on rus_rfe dataset...\n")
set.seed(1023)
rpart_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "rpart")

cat("\nTraining RPART model on rus_rf dataset...\n")
set.seed(1023)
rpart_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_rus_rf, test_data_rus_rf, "rus_rf", "rpart")

cat("\nTraining RPART model on rus_boruta dataset...\n")
set.seed(1023)
rpart_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "rpart")

cat("\nTraining RPART model on rus_ig dataset...\n")
set.seed(1023)
rpart_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_rus_ig, test_data_rus_ig, "rus_ig", "rpart")

cat("\nTraining RPART model on rus_chi_sq dataset...\n")
set.seed(1023)
rpart_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "rpart")

cat("\nTraining RPART model on smote_rfe dataset...\n")
set.seed(1023)
rpart_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "rpart")

cat("\nTraining RPART model on smote_rf dataset...\n")
set.seed(1023)
rpart_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_smote_rf, test_data_smote_rf, "smote_rf", "rpart")

cat("\nTraining RPART model on smote_boruta dataset...\n")
set.seed(1023)
rpart_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "rpart")

cat("\nTraining RPART model on smote_ig dataset...\n")
set.seed(1023)
rpart_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_smote_ig, test_data_smote_ig, "smote_ig", "rpart")

cat("\nTraining RPART model on smote_chi_sq dataset...\n")
set.seed(1023)
rpart_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "rpart", trControl = control, metric = "ROC")
evaluate_and_save_results(rpart_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "rpart")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

################################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Random Forest Models
cat("\nTraining Random forest model on ros_rfe dataset...\n")
set.seed(1023)
rf_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "rf")

cat("\nTraining Random forest model on ros_rf dataset...\n")
set.seed(1023)
rf_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_ros_rf, test_data_ros_rf, "ros_rf", "rf")

cat("\nTraining Random forest model on ros_boruta dataset...\n")
set.seed(1023)
rf_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "rf")

cat("\nTraining Random forest model on ros_ig dataset...\n")
set.seed(1023)
rf_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_ros_ig, test_data_ros_ig, "ros_ig", "rf")

cat("\nTraining Random forest model on ros_chi_sq dataset...\n")
set.seed(1023)
rf_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "rf")

cat("\nTraining Random forest model on rus_rfe dataset...\n")
set.seed(1023)
rf_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "rf")

cat("\nTraining Random forest model on rus_rf dataset...\n")
set.seed(1023)
rf_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_rus_rf, test_data_rus_rf, "rus_rf", "rf")

cat("\nTraining Random forest model on rus_boruta dataset...\n")
set.seed(1023)
rf_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "rf")

cat("\nTraining Random forest model on rus_ig dataset...\n")
set.seed(1023)
rf_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_rus_ig, test_data_rus_ig, "rus_ig", "rf")

cat("\nTraining Random forest model on rus_chi_sq dataset...\n")
set.seed(1023)
rf_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "rf")

cat("\nTraining Random forest model on smote_rfe dataset...\n")
set.seed(1023)
rf_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "rf")

cat("\nTraining Random forest model on smote_rf dataset...\n")
set.seed(1023)
rf_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_smote_rf, test_data_smote_rf, "smote_rf", "rf")

cat("\nTraining Random forest model on smote_boruta dataset...\n")
set.seed(1023)
rf_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "rf")

cat("\nTraining Random forest model on smote_ig dataset...\n")
set.seed(1023)
rf_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_smote_ig, test_data_smote_ig, "smote_ig", "rf")

cat("\nTraining Random forest model on smote_chi_sq dataset...\n")
set.seed(1023)
rf_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "rf", trControl = control, metric = "ROC")
evaluate_and_save_results(rf_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "rf")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()



################################################################################

# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# SVM Models
cat("\nTraining SVM model on ros_rfe dataset...\n")
set.seed(1023)
svm_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "svm")

cat("\nTraining SVM model on ros_rf dataset...\n")
set.seed(1023)
svm_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_ros_rf, test_data_ros_rf, "ros_rf", "svm")

cat("\nTraining SVM model on ros_boruta dataset...\n")
set.seed(1023)
svm_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "svm")

cat("\nTraining SVM model on ros_ig dataset...\n")
set.seed(1023)
svm_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_ros_ig, test_data_ros_ig, "ros_ig", "svm")

cat("\nTraining SVM model on ros_chi_sq dataset...\n")
set.seed(1023)
svm_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "svm")

cat("\nTraining SVM model on rus_rfe dataset...\n")
set.seed(1023)
svm_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "svm")

cat("\nTraining SVM model on rus_rf dataset...\n")
set.seed(1023)
svm_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_rus_rf, test_data_rus_rf, "rus_rf", "svm")

cat("\nTraining SVM model on rus_boruta dataset...\n")
set.seed(1023)
svm_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "svm")

cat("\nTraining SVM model on rus_ig dataset...\n")
set.seed(1023)
svm_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_rus_ig, test_data_rus_ig, "rus_ig", "svm")

cat("\nTraining SVM model on rus_chi_sq dataset...\n")
set.seed(1023)
svm_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "svm")

cat("\nTraining SVM model on smote_rfe dataset...\n")
set.seed(1023)
svm_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "svm")

cat("\nTraining SVM model on smote_rf dataset...\n")
set.seed(1023)
svm_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_smote_rf, test_data_smote_rf, "smote_rf", "svm")

cat("\nTraining SVM model on smote_boruta dataset...\n")
set.seed(1023)
svm_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "svm")

cat("\nTraining SVM model on smote_ig dataset...\n")
set.seed(1023)
svm_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_smote_ig, test_data_smote_ig, "smote_ig", "svm")

cat("\nTraining SVM model on smote_chi_sq dataset...\n")
set.seed(1023)
svm_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "svmRadial", trControl = control, metric = "ROC")
evaluate_and_save_results(svm_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "svm")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()


#################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Naive Bayes Models
cat("\nTraining Naive Bayes model on ros_rfe dataset...\n")
set.seed(1023)
nb_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "nb")

cat("\nTraining Naive Bayes model on ros_rf dataset...\n")
set.seed(1023)
nb_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_ros_rf, test_data_ros_rf, "ros_rf", "nb")

cat("\nTraining Naive Bayes model on ros_boruta dataset...\n")
set.seed(1023)
nb_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "nb")

cat("\nTraining Naive Bayes model on ros_ig dataset...\n")
set.seed(1023)
nb_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_ros_ig, test_data_ros_ig, "ros_ig", "nb")

cat("\nTraining Naive Bayes model on ros_chi_sq dataset...\n")
set.seed(1023)
nb_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "nb")

cat("\nTraining Naive Bayes model on rus_rfe dataset...\n")
set.seed(1023)
nb_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "nb")

cat("\nTraining Naive Bayes model on rus_rf dataset...\n")
set.seed(1023)
nb_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_rus_rf, test_data_rus_rf, "rus_rf", "nb")

cat("\nTraining Naive Bayes model on rus_boruta dataset...\n")
set.seed(1023)
nb_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "nb")

cat("\nTraining Naive Bayes model on rus_ig dataset...\n")
set.seed(1023)
nb_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_rus_ig, test_data_rus_ig, "rus_ig", "nb")

cat("\nTraining Naive Bayes model on rus_chi_sq dataset...\n")
set.seed(1023)
nb_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "nb")

cat("\nTraining Naive Bayes model on smote_rfe dataset...\n")
set.seed(1023)
nb_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "nb")

cat("\nTraining Naive Bayes model on smote_rf dataset...\n")
set.seed(1023)
nb_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_smote_rf, test_data_smote_rf, "smote_rf", "nb")

cat("\nTraining Naive Bayes model on smote_boruta dataset...\n")
set.seed(1023)
nb_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "nb")

cat("\nTraining Naive Bayes model on smote_ig dataset...\n")
set.seed(1023)
nb_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_smote_ig, test_data_smote_ig, "smote_ig", "nb")

cat("\nTraining Naive Bayes model on smote_chi_sq dataset...\n")
set.seed(1023)
nb_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "naive_bayes", trControl = control, metric = "ROC")
evaluate_and_save_results(nb_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "nb")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

###############################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# KNN Models
cat("\nTraining KNN model on ros_rfe dataset...\n")
set.seed(1023)
knn_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "knn")

cat("\nTraining KNN model on ros_rf dataset...\n")
set.seed(1023)
knn_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_ros_rf, test_data_ros_rf, "ros_rf", "knn")

cat("\nTraining KNN model on ros_boruta dataset...\n")
set.seed(1023)
knn_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "knn")

cat("\nTraining KNN model on ros_ig dataset...\n")
set.seed(1023)
knn_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_ros_ig, test_data_ros_ig, "ros_ig", "knn")

cat("\nTraining KNN model on ros_chi_sq dataset...\n")
set.seed(1023)
knn_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "knn")

cat("\nTraining KNN model on rus_rfe dataset...\n")
set.seed(1023)
knn_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "knn")

cat("\nTraining KNN model on rus_rf dataset...\n")
set.seed(1023)
knn_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_rus_rf, test_data_rus_rf, "rus_rf", "knn")

cat("\nTraining KNN model on rus_boruta dataset...\n")
set.seed(1023)
knn_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "knn")

cat("\nTraining KNN model on rus_ig dataset...\n")
set.seed(1023)
knn_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_rus_ig, test_data_rus_ig, "rus_ig", "knn")

cat("\nTraining KNN model on rus_chi_sq dataset...\n")
set.seed(1023)
knn_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "knn")

cat("\nTraining KNN model on smote_rfe dataset...\n")
set.seed(1023)
knn_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "knn")

cat("\nTraining KNN model on smote_rf dataset...\n")
set.seed(1023)
knn_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_smote_rf, test_data_smote_rf, "smote_rf", "knn")

cat("\nTraining KNN model on smote_boruta dataset...\n")
set.seed(1023)
knn_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "knn")

cat("\nTraining KNN model on smote_ig dataset...\n")
set.seed(1023)
knn_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_smote_ig, test_data_smote_ig, "smote_ig", "knn")

cat("\nTraining KNN model on smote_chi_sq dataset...\n")
set.seed(1023)
knn_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "knn", trControl = control, metric = "ROC")
evaluate_and_save_results(knn_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "knn")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()


################################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# GBM Models
cat("\nTraining GBM model on ros_rfe dataset...\n")
set.seed(1023)
gbm_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "gbm")

cat("\nTraining GBM model on ros_rf dataset...\n")
set.seed(1023)
gbm_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_ros_rf, test_data_ros_rf, "ros_rf", "gbm")

cat("\nTraining GBM model on ros_ig dataset...\n")
set.seed(1023)
gbm_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_ros_ig, test_data_ros_ig, "ros_ig", "gbm")

cat("\nTraining GBM model on ros_boruta dataset...\n")
set.seed(1023)
gbm_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "gbm")

cat("\nTraining GBM model on ros_chi_sq dataset...\n")
set.seed(1023)
gbm_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "gbm")

cat("\nTraining GBM model on rus_rfe dataset...\n")
set.seed(1023)
gbm_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "gbm")

cat("\nTraining GBM model on rus_rf dataset...\n")
set.seed(1023)
gbm_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_rus_rf, test_data_rus_rf, "rus_rf", "gbm")

cat("\nTraining GBM model on rus_ig dataset...\n")
set.seed(1023)
gbm_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_rus_ig, test_data_rus_ig, "rus_ig", "gbm")

cat("\nTraining GBM model on rus_boruta dataset...\n")
set.seed(1023)
gbm_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "gbm")

cat("\nTraining GBM model on rus_chi_sq dataset...\n")
set.seed(1023)
gbm_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "gbm")

cat("\nTraining GBM model on smote_rfe dataset...\n")
set.seed(1023)
gbm_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "gbm")

cat("\nTraining GBM model on smote_rf dataset...\n")
set.seed(1023)
gbm_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_smote_rf, test_data_smote_rf, "smote_rf", "gbm")

cat("\nTraining GBM model on smote_boruta dataset...\n")
set.seed(1023)
gbm_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "gbm")

cat("\nTraining GBM model on smote_ig dataset...\n")
set.seed(1023)
gbm_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_smote_ig, test_data_smote_ig, "smote_ig", "gbm")

cat("\nTraining GBM model on smote_chi_sq dataset...\n")
set.seed(1023)
gbm_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "gbm", trControl = control, metric = "ROC")
evaluate_and_save_results(gbm_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "gbm")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()


###############################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# XGBoost Models
cat("\nTraining XGBoost model on ros_rfe dataset...\n")
set.seed(1023)
xgb_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "xgboost")

cat("\nTraining XGBoost model on ros_rf dataset...\n")
set.seed(1023)
xgb_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_ros_rf, test_data_ros_rf, "ros_rf", "xgboost")

cat("\nTraining XGBoost model on ros_ig dataset...\n")
set.seed(1023)
xgb_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_ros_ig, test_data_ros_ig, "ros_ig", "xgboost")

cat("\nTraining XGBoost model on ros_boruta dataset...\n")
set.seed(1023)
xgb_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "xgboost")

cat("\nTraining XGBoost model on ros_chi_sq dataset...\n")
set.seed(1023)
xgb_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "xgboost")

cat("\nTraining XGBoost model on rus_rfe dataset...\n")
set.seed(1023)
xgb_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "xgboost")

cat("\nTraining XGBoost model on rus_rf dataset...\n")
set.seed(1023)
xgb_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_rus_rf, test_data_rus_rf, "rus_rf", "xgboost")

cat("\nTraining XGBoost model on rus_ig dataset...\n")
set.seed(1023)
xgb_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_rus_ig, test_data_rus_ig, "rus_ig", "xgboost")

cat("\nTraining XGBoost model on rus_boruta dataset...\n")
set.seed(1023)
xgb_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "xgboost")

cat("\nTraining XGBoost model on rus_chi_sq dataset...\n")
set.seed(1023)
xgb_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "xgboost")

cat("\nTraining XGBoost model on smote_rfe dataset...\n")
set.seed(1023)
xgb_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "xgboost")

cat("\nTraining XGBoost model on smote_rf dataset...\n")
set.seed(1023)
xgb_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_smote_rf, test_data_smote_rf, "smote_rf", "xgboost")

cat("\nTraining XGBoost model on smote_boruta dataset...\n")
set.seed(1023)
xgb_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "xgboost")

cat("\nTraining XGBoost model on smote_ig dataset...\n")
set.seed(1023)
xgb_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_smote_ig, test_data_smote_ig, "smote_ig", "xgboost")

cat("\nTraining XGBoost model on smote_chi_sq dataset...\n")
set.seed(1023)
xgb_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "xgbTree", trControl = control, metric = "ROC")
evaluate_and_save_results(xgb_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "xgboost")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()


################################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# AdaBoost Models
cat("\nTraining AdaBoost model on ros_rfe dataset...\n")
set.seed(1023)
ada_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "adaboost")

cat("\nTraining AdaBoost model on ros_rf dataset...\n")
set.seed(1023)
ada_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_ros_rf, test_data_ros_rf, "ros_rf", "adaboost")

cat("\nTraining AdaBoost model on ros_ig dataset...\n")
set.seed(1023)
ada_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_ros_ig, test_data_ros_ig, "ros_ig", "adaboost")

cat("\nTraining AdaBoost model on ros_boruta dataset...\n")
set.seed(1023)
ada_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "adaboost")

cat("\nTraining AdaBoost model on ros_chi_sq dataset...\n")
set.seed(1023)
ada_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "adaboost")

cat("\nTraining AdaBoost model on rus_rfe dataset...\n")
set.seed(1023)
ada_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "adaboost")

cat("\nTraining AdaBoost model on rus_rf dataset...\n")
set.seed(1023)
ada_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_rus_rf, test_data_rus_rf, "rus_rf", "adaboost")

cat("\nTraining AdaBoost model on rus_ig dataset...\n")
set.seed(1023)
ada_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_rus_ig, test_data_rus_ig, "rus_ig", "adaboost")

cat("\nTraining AdaBoost model on rus_boruta dataset...\n")
set.seed(1023)
ada_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "adaboost")

cat("\nTraining AdaBoost model on rus_chi_sq dataset...\n")
set.seed(1023)
ada_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "adaboost")

cat("\nTraining AdaBoost model on smote_rfe dataset...\n")
set.seed(1023)
ada_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "adaboost")

cat("\nTraining AdaBoost model on smote_rf dataset...\n")
set.seed(1023)
ada_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_smote_rf, test_data_smote_rf, "smote_rf", "adaboost")

cat("\nTraining AdaBoost model on smote_boruta dataset...\n")
set.seed(1023)
ada_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "adaboost")

cat("\nTraining AdaBoost model on smote_ig dataset...\n")
set.seed(1023)
ada_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_smote_ig, test_data_smote_ig, "smote_ig", "adaboost")

cat("\nTraining AdaBoost model on smote_chi_sq dataset...\n")
set.seed(1023)
ada_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "AdaBoost.M1", trControl = control, metric = "ROC")
evaluate_and_save_results(ada_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "adaboost")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()


############################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# MLP Models
cat("\nTraining MLP model on ros_rfe dataset...\n")
set.seed(1023)
mlp_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "mlp")

cat("\nTraining MLP model on ros_rf dataset...\n")
set.seed(1023)
mlp_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_ros_rf, test_data_ros_rf, "ros_rf", "mlp")

cat("\nTraining MLP model on ros_ig dataset...\n")
set.seed(1023)
mlp_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_ros_ig, test_data_ros_ig, "ros_ig", "mlp")

cat("\nTraining MLP model on ros_boruta dataset...\n")
set.seed(1023)
mlp_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "mlp")

cat("\nTraining MLP model on ros_chi_sq dataset...\n")
set.seed(1023)
mlp_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "mlp")

cat("\nTraining MLP model on rus_rfe dataset...\n")
set.seed(1023)
mlp_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "mlp")

cat("\nTraining MLP model on rus_rf dataset...\n")
set.seed(1023)
mlp_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_rus_rf, test_data_rus_rf, "rus_rf", "mlp")

cat("\nTraining MLP model on rus_ig dataset...\n")
set.seed(1023)
mlp_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_rus_ig, test_data_rus_ig, "rus_ig", "mlp")

cat("\nTraining MLP model on rus_boruta dataset...\n")
set.seed(1023)
mlp_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "mlp")

cat("\nTraining MLP model on rus_chi_sq dataset...\n")
set.seed(1023)
mlp_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "mlp")

cat("\nTraining MLP model on smote_rfe dataset...\n")
set.seed(1023)
mlp_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "mlp")

cat("\nTraining MLP model on smote_rf dataset...\n")
set.seed(1023)
mlp_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_smote_rf, test_data_smote_rf, "smote_rf", "mlp")

cat("\nTraining MLP model on smote_boruta dataset...\n")
set.seed(1023)
mlp_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "mlp")

cat("\nTraining MLP model on smote_ig dataset...\n")
set.seed(1023)
mlp_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_smote_ig, test_data_smote_ig, "smote_ig", "mlp")

cat("\nTraining MLP model on smote_chi_sq dataset...\n")
set.seed(1023)
mlp_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "mlp", trControl = control, metric = "ROC")
evaluate_and_save_results(mlp_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "mlp")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()


###############################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Extreme Trees Models (ranger)
cat("\nTraining Extreme Trees model on ros_rfe dataset...\n")
set.seed(1023)
extra_trees_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "extraTrees")

cat("\nTraining Extreme Trees model on ros_rf dataset...\n")
set.seed(1023)
extra_trees_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_rf, test_data_ros_rf, "ros_rf", "extraTrees")

cat("\nTraining Extreme Trees model on ros_ig dataset...\n")
set.seed(1023)
extra_trees_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_ig, test_data_ros_ig, "ros_ig", "extraTrees")

cat("\nTraining Extreme Trees model on ros_boruta dataset...\n")
set.seed(1023)
extra_trees_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "extraTrees")

cat("\nTraining Extreme Trees model on ros_chi_sq dataset...\n")
set.seed(1023)
extra_trees_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "extraTrees")

cat("\nTraining Extreme Trees model on rus_rfe dataset...\n")
set.seed(1023)
extra_trees_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "extraTrees")

cat("\nTraining Extreme Trees model on rus_rf dataset...\n")
set.seed(1023)
extra_trees_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_rf, test_data_rus_rf, "rus_rf", "extraTrees")

cat("\nTraining Extreme Trees model on rus_ig dataset...\n")
set.seed(1023)
extra_trees_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_ig, test_data_rus_ig, "rus_ig", "extraTrees")

cat("\nTraining Extreme Trees model on rus_boruta dataset...\n")
set.seed(1023)
extra_trees_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "extraTrees")

cat("\nTraining Extreme Trees model on rus_chi_sq dataset...\n")
set.seed(1023)
extra_trees_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "extraTrees")

cat("\nTraining Extreme Trees model on smote_rfe dataset...\n")
set.seed(1023)
extra_trees_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "extraTrees")

cat("\nTraining Extreme Trees model on smote_rf dataset...\n")
set.seed(1023)
extra_trees_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_rf, test_data_smote_rf, "smote_rf", "extraTrees")

cat("\nTraining Extreme Trees model on smote_boruta dataset...\n")
set.seed(1023)
extra_trees_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "extraTrees")

cat("\nTraining Extreme Trees model on smote_ig dataset...\n")
set.seed(1023)
extra_trees_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_ig, test_data_smote_ig, "smote_ig", "extraTrees")

cat("\nTraining Extreme Trees model on smote_chi_sq dataset...\n")
set.seed(1023)
extra_trees_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "extraTrees")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

###############################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
#hyperparamenter tunning efforts

# Define grids for hyperparameter tuning
tuneGrids <- list(
  #we will setup elastic net for GLM tunning
  rpart = expand.grid(cp = seq(0.001, 0.2, by = 0.005)),
  rf = expand.grid(mtry = c(2, 5, 7, 10, 15, 20)),
  nb = expand.grid(laplace = c(0, 0.5, 1, 1.5), usekernel = c(TRUE, FALSE), adjust = c(0.5, 1, 1.5, 2)),
  kknn = expand.grid(kmax = c(3, 5, 7, 9, 11), distance = c(1, 2), kernel = c("rectangular", "triangular", "epanechnikov", "optimal")),
  svmRadial = expand.grid(C = c(0.01, 0.1, 1, 10, 100), sigma = c(0.001, 0.01, 0.1, 1))
)

# Create the tuning grid for Elastic Net
tune_grid_elastic_net <- expand.grid(
  alpha = seq(0, 1, by = 0.1),  # Alpha values from 0 to 1 in increments of 0.1
  lambda = 10^seq(-4, 1, length = 10)  # Lambda values from 10^-4 to 10^1
)

#GBM Tuning Grid
gbm_grid <- expand.grid(
  n.trees = c(50, 100, 150),
  interaction.depth = c(1, 3, 5, 7),
  shrinkage = c(0.01, 0.05, 0.1, 0.3),
  n.minobsinnode = c(10, 20)
)

xgb_grid <- expand.grid(
  nrounds = c(50, 100),
  max_depth = c(3, 6),
  eta = c(0.01, 0.1),
  gamma = c(0, 1),
  colsample_bytree = c(0.7, 1),
  min_child_weight = c(1, 3),
  subsample = c(0.7, 1)
)

# AdaBoost Tuning Grid
ada_grid <- expand.grid(
  mfinal = c(50, 100, 150),   
  maxdepth = c(1, 3, 5),      
  coeflearn = c("Breiman", "Freund", "Zhu") #learning rate 
)

# Define a more strict tuning grid 
mlp_grid <- expand.grid(size = c(3, 5, 7))

#grid for extreme tree is defined in the models

##############################################################################
#train the models with hyperparameter tunning
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
# Control for cross-validation
control <- trainControl(method = "cv", number = 10, repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)

# Training and evaluation for each dataset using Elastic Net regularization (GLM)
cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on ros_rfe dataset...\n")
set.seed(1023)
glmnet_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on ros_rf dataset...\n")
set.seed(1023)
glmnet_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_ros_rf, test_data_ros_rf, "ros_rf", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on ros_boruta dataset...\n")
set.seed(1023)
glmnet_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on ros_ig dataset...\n")
set.seed(1023)
glmnet_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_ros_ig, test_data_ros_ig, "ros_ig", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on ros_chi_sq dataset...\n")
set.seed(1023)
glmnet_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on rus_rfe dataset...\n")
set.seed(1023)
glmnet_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on rus_rf dataset...\n")
set.seed(1023)
glmnet_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_rus_rf, test_data_rus_rf, "rus_rf", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on rus_boruta dataset...\n")
set.seed(1023)
glmnet_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on rus_ig dataset...\n")
set.seed(1023)
glmnet_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_rus_ig, test_data_rus_ig, "rus_ig", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on rus_chi_sq dataset...\n")
set.seed(1023)
glmnet_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on smote_rfe dataset...\n")
set.seed(1023)
glmnet_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on smote_rf dataset...\n")
set.seed(1023)
glmnet_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_smote_rf, test_data_smote_rf, "smote_rf", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on smote_boruta dataset...\n")
set.seed(1023)
glmnet_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on smote_ig dataset...\n")
set.seed(1023)
glmnet_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_smote_ig, test_data_smote_ig, "smote_ig", "glmnet_hyper")

cat("\nTraining GLM (Elastic Net) model with hyperparameter tuning on smote_chi_sq dataset...\n")
set.seed(1023)
glmnet_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "glmnet", trControl = control, metric = "ROC", tuneGrid = tune_grid_elastic_net)
evaluate_and_save_results(glmnet_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "glmnet_hyper")

################################################################################
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
# Training and evaluation for each dataset using RPART with hyperparameter tuning
cat("\nTraining RPART (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
rpart_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
rpart_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_ros_rf, test_data_ros_rf, "ros_rf", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
rpart_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
rpart_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_ros_ig, test_data_ros_ig, "ros_ig", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
rpart_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
rpart_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
rpart_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_rus_rf, test_data_rus_rf, "rus_rf", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
rpart_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
rpart_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_rus_ig, test_data_rus_ig, "rus_ig", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
rpart_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
rpart_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
rpart_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_smote_rf, test_data_smote_rf, "smote_rf", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
rpart_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
rpart_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_smote_ig, test_data_smote_ig, "smote_ig", "rpart_hyper")

cat("\nTraining RPART (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
rpart_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "rpart", trControl = control, metric = "ROC", tuneGrid = tuneGrids$rpart)
evaluate_and_save_results(rpart_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "rpart_hyper")


#######################################################################################

# Set up the RF tuning grid
rf_grid <- tuneGrids$rf
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
# Training and evaluation for each dataset using RF with hyperparameter tuning
cat("\nTraining RF (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
rf_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
rf_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_ros_rf, test_data_ros_rf, "ros_rf", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
rf_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
rf_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_ros_ig, test_data_ros_ig, "ros_ig", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
rf_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
rf_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
rf_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_rus_rf, test_data_rus_rf, "rus_rf", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
rf_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
rf_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_rus_ig, test_data_rus_ig, "rus_ig", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
rf_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
rf_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
rf_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_smote_rf, test_data_smote_rf, "smote_rf", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
rf_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
rf_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_smote_ig, test_data_smote_ig, "smote_ig", "rf_hyper")

cat("\nTraining RF (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
rf_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "rf", trControl = control, metric = "ROC", tuneGrid = rf_grid)
evaluate_and_save_results(rf_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "rf_hyper")

################################################################################

# Set up the NB tuning grid
nb_grid <- tuneGrids$nb

# Training and evaluation for each dataset using NB with hyperparameter tuning
cat("\nTraining NB (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
nb_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
nb_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_ros_rf, test_data_ros_rf, "ros_rf", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
nb_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
nb_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_ros_ig, test_data_ros_ig, "ros_ig", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
nb_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
nb_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
nb_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_rus_rf, test_data_rus_rf, "rus_rf", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
nb_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
nb_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_rus_ig, test_data_rus_ig, "rus_ig", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
nb_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
nb_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
nb_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_smote_rf, test_data_smote_rf, "smote_rf", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
nb_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
nb_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_smote_ig, test_data_smote_ig, "smote_ig", "nb_hyper")

cat("\nTraining NB (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
nb_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "naive_bayes", trControl = control, metric = "ROC", tuneGrid = nb_grid)
evaluate_and_save_results(nb_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "nb_hyper")

#################################################################################

# Set up the KNN tuning grid
knn_grid <- tuneGrids$kknn

# Training and evaluation for each dataset using KNN with hyperparameter tuning
cat("\nTraining KNN (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
knn_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
knn_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_ros_rf, test_data_ros_rf, "ros_rf", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
knn_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
knn_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_ros_ig, test_data_ros_ig, "ros_ig", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
knn_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
knn_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
knn_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_rus_rf, test_data_rus_rf, "rus_rf", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
knn_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
knn_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_rus_ig, test_data_rus_ig, "rus_ig", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
knn_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
knn_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
knn_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_smote_rf, test_data_smote_rf, "smote_rf", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
knn_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
knn_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_smote_ig, test_data_smote_ig, "smote_ig", "knn_hyper")

cat("\nTraining KNN (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
knn_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "kknn", trControl = control, metric = "ROC", tuneGrid = knn_grid)
evaluate_and_save_results(knn_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "knn_hyper")

################################################################################

# Set up the SVM tuning grid
svm_grid <- tuneGrids$svmRadial

# Training and evaluation for each dataset using SVM with hyperparameter tuning
cat("\nTraining SVM (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
svm_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
svm_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_ros_rf, test_data_ros_rf, "ros_rf", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
svm_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
svm_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_ros_ig, test_data_ros_ig, "ros_ig", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
svm_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
svm_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
svm_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_rus_rf, test_data_rus_rf, "rus_rf", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
svm_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
svm_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_rus_ig, test_data_rus_ig, "rus_ig", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
svm_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
svm_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
svm_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_smote_rf, test_data_smote_rf, "smote_rf", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
svm_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
svm_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_smote_ig, test_data_smote_ig, "smote_ig", "svm_hyper")

cat("\nTraining SVM (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
svm_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "svmRadial", trControl = control, metric = "ROC", tuneGrid = svm_grid)
evaluate_and_save_results(svm_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "svm_hyper")

################################################################################

# Training and evaluation for each dataset using GBM with hyperparameter tuning
cat("\nTraining GBM (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
gbm_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
gbm_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_ros_rf, test_data_ros_rf, "ros_rf", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
gbm_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
gbm_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_ros_ig, test_data_ros_ig, "ros_ig", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
gbm_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
gbm_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
gbm_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_rus_rf, test_data_rus_rf, "rus_rf", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
gbm_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
gbm_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_rus_ig, test_data_rus_ig, "rus_ig", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
gbm_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
gbm_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
gbm_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_smote_rf, test_data_smote_rf, "smote_rf", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
gbm_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
gbm_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_smote_ig, test_data_smote_ig, "smote_ig", "gbm_hyper")

cat("\nTraining GBM (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
gbm_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "gbm", trControl = control, metric = "ROC", tuneGrid = gbm_grid)
evaluate_and_save_results(gbm_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "gbm_hyper")

#################################################################################

# Training and evaluation for each dataset using XGBoost with hyperparameter tuning
cat("\nTraining XGBoost (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
xgb_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
xgb_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_ros_rf, test_data_ros_rf, "ros_rf", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
xgb_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
xgb_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_ros_ig, test_data_ros_ig, "ros_ig", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
xgb_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
xgb_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
xgb_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_rus_rf, test_data_rus_rf, "rus_rf", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
xgb_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
xgb_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_rus_ig, test_data_rus_ig, "rus_ig", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
xgb_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
xgb_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
xgb_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_smote_rf, test_data_smote_rf, "smote_rf", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
xgb_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
xgb_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_smote_ig, test_data_smote_ig, "smote_ig", "xgb_hyper")

cat("\nTraining XGBoost (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
xgb_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "xgbTree", trControl = control, metric = "ROC", tuneGrid = xgb_grid)
evaluate_and_save_results(xgb_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "xgb_hyper")

################################################################################

# Training and evaluation for each dataset using AdaBoost with hyperparameter tuning
cat("\nTraining AdaBoost (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
ada_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
ada_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_ros_rf, test_data_ros_rf, "ros_rf", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
ada_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
ada_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_ros_ig, test_data_ros_ig, "ros_ig", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
ada_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
ada_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
ada_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_rus_rf, test_data_rus_rf, "rus_rf", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
ada_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
ada_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_rus_ig, test_data_rus_ig, "rus_ig", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
ada_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
ada_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
ada_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_smote_rf, test_data_smote_rf, "smote_rf", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
ada_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
ada_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_smote_ig, test_data_smote_ig, "smote_ig", "adaboost_hyper")

cat("\nTraining AdaBoost (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
ada_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "AdaBoost.M1", trControl = control, metric = "ROC", tuneGrid = ada_grid)
evaluate_and_save_results(ada_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "adaboost_hyper")

################################################################################

# Training and evaluation for each dataset using MLP with hyperparameter tuning
cat("\nTraining MLP (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
mlp_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
mlp_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_ros_rf, test_data_ros_rf, "ros_rf", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
mlp_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
mlp_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_ros_ig, test_data_ros_ig, "ros_ig", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
mlp_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
mlp_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
mlp_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_rus_rf, test_data_rus_rf, "rus_rf", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
mlp_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
mlp_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_rus_ig, test_data_rus_ig, "rus_ig", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
mlp_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
mlp_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
mlp_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_smote_rf, test_data_smote_rf, "smote_rf", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
mlp_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
mlp_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_smote_ig, test_data_smote_ig, "smote_ig", "mlp_hyper")

cat("\nTraining MLP (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
mlp_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "mlp", trControl = control, metric = "ROC", tuneGrid = mlp_grid)
evaluate_and_save_results(mlp_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "mlp_hyper")

################################################################################

# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
# Training and evaluation for each dataset using Extremely Randomized Trees with hyperparameter tuning
cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on ros_rfe dataset...\n")
set.seed(1023)
extra_trees_model_ros_rfe <- caret::train(Class ~ ., data = feature_datasets$ros_rfe, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_rfe, test_data_ros_rfe, "ros_rfe", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on ros_rf dataset...\n")
set.seed(1023)
extra_trees_model_ros_rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_rf, test_data_ros_rf, "ros_rf", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on ros_boruta dataset...\n")
set.seed(1023)
extra_trees_model_ros_boruta <- caret::train(Class ~ ., data = feature_datasets$ros_boruta, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_boruta, test_data_ros_boruta, "ros_boruta", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on ros_ig dataset...\n")
set.seed(1023)
extra_trees_model_ros_ig <- caret::train(Class ~ ., data = feature_datasets$ros_ig, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_ig, test_data_ros_ig, "ros_ig", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on ros_chi_sq dataset...\n")
set.seed(1023)
extra_trees_model_ros_chi_sq <- caret::train(Class ~ ., data = feature_datasets$ros_chi_sq, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_ros_chi_sq, test_data_ros_chi_sq, "ros_chi_sq", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on rus_rfe dataset...\n")
set.seed(1023)
extra_trees_model_rus_rfe <- caret::train(Class ~ ., data = feature_datasets$rus_rfe, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_rfe, test_data_rus_rfe, "rus_rfe", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on rus_rf dataset...\n")
set.seed(1023)
extra_trees_model_rus_rf <- caret::train(Class ~ ., data = feature_datasets$rus_rf, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_rf, test_data_rus_rf, "rus_rf", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on rus_boruta dataset...\n")
set.seed(1023)
extra_trees_model_rus_boruta <- caret::train(Class ~ ., data = feature_datasets$rus_boruta, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_boruta, test_data_rus_boruta, "rus_boruta", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on rus_ig dataset...\n")
set.seed(1023)
extra_trees_model_rus_ig <- caret::train(Class ~ ., data = feature_datasets$rus_ig, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_ig, test_data_rus_ig, "rus_ig", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on rus_chi_sq dataset...\n")
set.seed(1023)
extra_trees_model_rus_chi_sq <- caret::train(Class ~ ., data = feature_datasets$rus_chi_sq, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_rus_chi_sq, test_data_rus_chi_sq, "rus_chi_sq", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on smote_rfe dataset...\n")
set.seed(1023)
extra_trees_model_smote_rfe <- caret::train(Class ~ ., data = feature_datasets$smote_rfe, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_rfe, test_data_smote_rfe, "smote_rfe", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on smote_rf dataset...\n")
set.seed(1023)
extra_trees_model_smote_rf <- caret::train(Class ~ ., data = feature_datasets$smote_rf, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_rf, test_data_smote_rf, "smote_rf", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on smote_boruta dataset...\n")
set.seed(1023)
extra_trees_model_smote_boruta <- caret::train(Class ~ ., data = feature_datasets$smote_boruta, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_boruta, test_data_smote_boruta, "smote_boruta", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on smote_ig dataset...\n")
set.seed(1023)
extra_trees_model_smote_ig <- caret::train(Class ~ ., data = feature_datasets$smote_ig, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_ig, test_data_smote_ig, "smote_ig", "extraTrees_hyper")

cat("\nTraining Extremely Randomized Trees (hyperparameter tuned) model on smote_chi_sq dataset...\n")
set.seed(1023)
extra_trees_model_smote_chi_sq <- caret::train(Class ~ ., data = feature_datasets$smote_chi_sq, method = "ranger", trControl = control, metric = "ROC", tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
evaluate_and_save_results(extra_trees_model_smote_chi_sq, test_data_smote_chi_sq, "smote_chi_sq", "extraTrees_hyper")

################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################


############Ensemble Method#################
# Define the list to store models
models_list <- list()
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Train all models
set.seed(1023)
models_list$rf <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "rf", trControl = control, tuneGrid = tuneGrids$rf)
models_list$svm <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "svmRadial", trControl = control, tuneGrid = tuneGrids$svmRadial)
models_list$extra_trees <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "ranger", trControl = control, tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 1))
models_list$gbm <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "gbm", trControl = control, tuneGrid = gbm_grid)
models_list$mlp <- mlp_model_ros_rf
models_list$glm <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "glm", trControl = control)
models_list$ada <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "AdaBoost.M1", trControl = control, tuneGrid = ada_grid)
models_list$nb <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "naive_bayes", trControl = control, tuneGrid = tuneGrids$nb)
models_list$knn <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "kknn", trControl = control, tuneGrid = tuneGrids$kknn)
models_list$xgb <- caret::train(Class ~ ., data = feature_datasets$ros_rf, method = "xgbTree", trControl = control, tuneGrid = xgb_grid)



# Generate predictions for the test data and calculate errors
predictions_list <- list()
errors_list <- list()

for (model_name in names(models_list)) {
  preds <- predict(models_list[[model_name]], test_data_ros_rf, type = "prob")[,2]
  predictions_list[[model_name]] <- preds
  errors_list[[model_name]] <- as.numeric(test_data_ros_rf$Class) - preds
}
# Compute error correlation matrix
errors_matrix <- do.call(cbind, errors_list)
colnames(errors_matrix) <- names(errors_list)
error_correlation_matrix <- cor(errors_matrix)
print(error_correlation_matrix)

# Visualize the error correlation matrix
library(corrplot)
corrplot(error_correlation_matrix, method = "circle", type = "upper")

# Function to compute prediction errors
compute_errors <- function(model, test_data) {
  predictions <- predict(model, newdata = test_data)
  errors <- ifelse(predictions == test_data$Class, 0, 1)
  return(errors)
}

# Compute prediction errors for each model
errors <- lapply(models_list, compute_errors, test_data = test_data_ros_rf)

# Convert the list of errors to a data frame
errors_df <- as.data.frame(errors)

# Compute the correlation matrix
error_correlation_matrix <- cor(errors_df)

# Print the correlation matrix
print(error_correlation_matrix)


# Train base models on ros_rf dataset
train_base_models <- function(dataset, control) {
  print("Starting to train base models...")
  
  print("Training SVM...")
  svm_model <- caret::train(Class ~ ., data = dataset, method = "svmRadial", trControl = control, tuneGrid = tuneGrids$svmRadial, metric = "ROC")
  
  print("Training MLP...")
  mlp_model <- caret::train(Class ~ ., data = dataset, method = "mlp", trControl = control, tuneGrid = mlp_grid, metric = "ROC")
  
  print("Training Random Forest...")
  rf_model <- caret::train(Class ~ ., data = dataset, method = "rf", trControl = control, tuneGrid = tuneGrids$rf, metric = "ROC")
  
  print("Training AdaBoost...")
  ada_model <- caret::train(Class ~ ., data = dataset, method = "AdaBoost.M1", trControl = control, tuneGrid = ada_grid, metric = "ROC")
  
  print("Training knn...")
  knn_model <- caret::train(Class ~ ., data = dataset, method = "kknn", trControl = control, metric = "ROC",tuneGrid = tuneGrids$kknn)
  print("Combining models into list...")
  base_models <- list(
    svm = svm_model,
    mlp = mlp_model,
    rf = rf_model,
    ada = ada_model,
    knn = knn_model
  )
  
  print("Base models training complete.")
  return(base_models)
}

# Generate base model predictions
generate_base_predictions <- function(base_models, dataset) {
  base_predictions <- as.data.frame(lapply(base_models, function(model) predict(model, dataset, type = "prob")[, 2]))
  base_predictions$Class <- dataset$Class
  return(base_predictions)
}

# Train meta-model
train_meta_model <- function(base_predictions, control) {
  meta_model <- caret::train(Class ~ ., data = base_predictions, method = "glm",family = binomial, trControl = control)
  return(meta_model)
}





# Function to calculate performance metrics
calculate_performance_metrics <- function(predictions, probabilities, actuals) {
  confusion <- caret::confusionMatrix(predictions, actuals)
  
  tpr0 <- caret::sensitivity(predictions, reference = actuals, positive = "Class0")
  fpr0 <- 1 - caret::specificity(predictions, reference = actuals, positive = "Class0")
  precision0 <- caret::posPredValue(predictions, reference = actuals, positive = "Class0")
  recall0 <- tpr0
  f_measure0 <- 2 * (precision0 * recall0) / (precision0 + recall0)
  
  tpr1 <- caret::sensitivity(predictions, reference = actuals, positive = "Class1")
  fpr1 <- 1 - caret::specificity(predictions, reference = actuals, positive = "Class1")
  precision1 <- caret::posPredValue(predictions, reference = actuals, positive = "Class1")
  recall1 <- tpr1
  f_measure1 <- 2 * (precision1 * recall1) / (precision1 + recall1)
  
  roc_curve <- pROC::roc(actuals, as.numeric(probabilities[, "Class1"]))
  auc <- pROC::auc(roc_curve)
  mcc <- calculate_mcc(confusion$table["Class1", "Class1"], confusion$table["Class0", "Class0"], 
                       confusion$table["Class1", "Class0"], confusion$table["Class0", "Class1"])
  kappa <- confusion$overall["Kappa"]
  
  weighted_avg <- function(metric_class0, metric_class1) {
    total <- length(actuals)
    weight_class0 <- sum(actuals == "Class0") / total
    weight_class1 <- sum(actuals == "Class1") / total
    weighted_metric <- (metric_class0 * weight_class0) + (metric_class1 * weight_class1)
    return(weighted_metric)
  }
  
  performance_table <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F_measure", "AUC", "MCC", "Kappa"),
    Class0 = c(tpr0, fpr0, precision0, recall0, f_measure0, auc, mcc, kappa),
    Class1 = c(tpr1, fpr1, precision1, recall1, f_measure1, auc, mcc, kappa),
    Weighted_Avg = c(
      weighted_avg(tpr0, tpr1), 
      weighted_avg(fpr0, fpr1), 
      weighted_avg(precision0, precision1), 
      weighted_avg(recall0, recall1), 
      weighted_avg(f_measure0, f_measure1),
      auc, # AUC is already an average
      mcc, # MCC is already an average
      kappa # Kappa is already an average
    )
  )
  
  # Transpose the data frame to get a horizontal layout
  performance_table_t <- t(performance_table)
  colnames(performance_table_t) <- performance_table$Metric
  performance_table_t <- performance_table_t[-1, ] # Remove the Metric row
  
  return(list(performance_table = performance_table_t, conf_matrix = confusion$table, roc_curve = roc_curve))
}

# Function to evaluate and save results for ensemble models
evaluate_and_save_results_ensemble <- function(predictions, probabilities, actuals, dataset_name, model_name) {
  evaluation <- calculate_performance_metrics(predictions, probabilities, actuals)
  performance_table <- evaluation$performance_table
  conf_matrix <- evaluation$conf_matrix
  roc_curve <- evaluation$roc_curve
  
  # Print the confusion matrix
  cat("\nConfusion Matrix for", model_name, "model on", dataset_name, "dataset:\n")
  print(conf_matrix)
  
  # Print the performance table
  cat("\nPerformance Table for", model_name, "model on", dataset_name, "dataset:\n")
  print(performance_table)
  
  # Save the performance table to a CSV file
  write.csv(performance_table, file = paste0("performance_metrics_ensemble_", model_name, "_", dataset_name, ".csv"), row.names = TRUE)
  
  # Save the confusion matrix to a CSV file
  write.csv(as.data.frame.matrix(conf_matrix), file = paste0("confusion_matrix_ensemble_", model_name, "_", dataset_name, ".csv"), row.names = TRUE)
  
  # Save the ROC plot
  png(filename = paste0("roc_curve_ensemble_", model_name, "_", dataset_name, ".png"))
  plot(roc_curve, main = paste("ROC Curve for", model_name, "on", dataset_name), col = "blue")
  abline(a = 0, b = 1, col = "gray", lty = 2)
  dev.off()
}




# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
set.seed(1023)
control <- trainControl(method = "repeatedcv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary, sampling = "smote")
# Train and evaluate ensemble model on ros_rf dataset
cat("\nTraining ensemble model on ros_rf dataset...\n")
dataset_name <- "ros_rf"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_ros_rf)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_ros_rf$Class, dataset_name, "ensemble")


# Train and evaluate ensemble model on ros_rfe dataset
cat("\nTraining ensemble model on ros_ref dataset...\n")
dataset_name <- "ros_rfe"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_ros_rf)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_ros_rfe$Class, dataset_name, "ensemble")


# Dataset 3: ros_boruta
cat("\nTraining ensemble model on ros_boruta dataset...\n")
dataset_name <- "ros_boruta"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_ros_boruta)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_ros_boruta$Class, dataset_name, "ensemble")

# Dataset 4: ros_ig
cat("\nTraining ensemble model on ros_ig dataset...\n")
dataset_name <- "ros_ig"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_ros_ig)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_ros_ig$Class, dataset_name, "ensemble")

# Dataset 5: ros_chi_sq
cat("\nTraining ensemble model on ros_chi_sq dataset...\n")
dataset_name <- "ros_chi_sq"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_ros_chi_sq)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_ros_chi_sq$Class, dataset_name, "ensemble")

# Dataset 6: rus_rfe
cat("\nTraining ensemble model on rus_rfe dataset...\n")
dataset_name <- "rus_rfe"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_rus_rfe)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_rus_rfe$Class, dataset_name, "ensemble")

# Dataset 7: rus_rf
cat("\nTraining ensemble model on rus_rf dataset...\n")
dataset_name <- "rus_rf"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_rus_rf)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_rus_rf$Class, dataset_name, "ensemble")

# Dataset 8: rus_boruta
cat("\nTraining ensemble model on rus_boruta dataset...\n")
dataset_name <- "rus_boruta"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_rus_boruta)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_rus_boruta$Class, dataset_name, "ensemble")

# Dataset 9: rus_ig
cat("\nTraining ensemble model on rus_ig dataset...\n")
dataset_name <- "rus_ig"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_rus_ig)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_rus_ig$Class, dataset_name, "ensemble")

# Dataset 10: rus_chi_sq
cat("\nTraining ensemble model on rus_chi_sq dataset...\n")
dataset_name <- "rus_chi_sq"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_rus_chi_sq)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_rus_chi_sq$Class, dataset_name, "ensemble")

# Dataset 11: smote_rfe
cat("\nTraining ensemble model on smote_rfe dataset...\n")
dataset_name <- "smote_rfe"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_smote_rfe)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_smote_rfe$Class, dataset_name, "ensemble")

# Dataset 12: smote_rf
cat("\nTraining ensemble model on smote_rf dataset...\n")
dataset_name <- "smote_rf"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_smote_rf)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_smote_rf$Class, dataset_name, "ensemble")

# Dataset 13: smote_ig
cat("\nTraining ensemble model on smote_ig dataset...\n")
dataset_name <- "smote_ig"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_smote_ig)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_smote_ig$Class, dataset_name, "ensemble")

# Dataset 14: smote_boruta
cat("\nTraining ensemble model on smote_boruta dataset...\n")
dataset_name <- "smote_boruta"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_smote_boruta)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_smote_boruta$Class, dataset_name, "ensemble")

# Dataset 15: smote_chi_sq
cat("\nTraining ensemble model on smote_chi_sq dataset...\n")
dataset_name <- "smote_chi_sq"
base_models <- train_base_models(feature_datasets[[dataset_name]], control)
train_predictions <- generate_base_predictions(base_models, feature_datasets[[dataset_name]])
meta_model <- train_meta_model(train_predictions, control)
test_predictions <- generate_base_predictions(base_models, test_data_smote_chi_sq)
meta_test_predictions <- predict(meta_model, test_predictions)
meta_test_probabilities <- predict(meta_model, test_predictions, type = "prob")
evaluate_and_save_results_ensemble(meta_test_predictions, meta_test_probabilities, test_data_smote_chi_sq$Class, dataset_name, "ensemble")
# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################