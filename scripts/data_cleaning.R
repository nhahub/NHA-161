# ============================================================================
# Customer Churn Data Cleaning (R)
# ============================================================================
# This script loads and cleans customer churn data.
# Steps:
# 1. Load required libraries
# 2. Load and clean data
# 3. Handle class imbalance with undersampling
# 4. Export cleaned datasets
# ============================================================================

# --- Configure Project Library Path ---
project_lib <- file.path(getwd(), ".r-lib")
if(!dir.exists(project_lib)) {
  dir.create(project_lib, recursive = TRUE)
}
.libPaths(c(project_lib, .libPaths()))

# --- Load Required Libraries ---
required_packages <- c("openxlsx", "caTools")
cat("Required packages:", paste(required_packages, collapse = ", "), "\n")

missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cloud.r-project.org", lib = project_lib, dependencies = TRUE)
}

invisible(lapply(required_packages, require, character.only = TRUE))

# --- Set Working Directory ---
setwd('C:\\Users\\AHNADXR\\Desktop\\EDUCATION C\\DEPI\\project\\1\\data')

# ============================================================================
# SECTION 1: DATA LOADING AND CLEANING
# ============================================================================

cat("Loading data...\n")
# Load data
churn <- read.xlsx('raw_data.xlsx', sheet = 1)
cat("Data loaded successfully!\n")
cat("Initial shape:", nrow(churn), "rows,", ncol(churn), "columns\n\n")

# Remove first two columns (likely ID columns)
churn <- churn[, -c(1, 2)]

# Check and remove duplicates
cat("Checking for duplicates...\n")
duplicate_count <- sum(duplicated(churn))
cat("Duplicates found:", duplicate_count, "\n")

if(duplicate_count > 0) {
  churn <- churn[!duplicated(churn), ]
  cat("Duplicates removed\n")
}

cat("Shape after removing duplicates:", nrow(churn), "rows,", ncol(churn), "columns\n\n")

# Check data structure
cat("Data structure:\n")
str(churn)

# Convert categorical columns to factors
cat("\nConverting categorical columns to factors...\n")
categorical_cols <- c('Product.Category', 'Payment.Method', 'Returns', 
                      'Customer.Name', 'Gender', 'Churn')

for(col in categorical_cols) {
  if(col %in% names(churn)) {
    churn[[col]] <- as.factor(churn[[col]])
  }
}

# Create binary churn target variable
cat("Creating binary target variable...\n")
churn$Churn_yes <- ifelse(churn$Churn == 'churned', 1, 0)
churn$Churn <- NULL

cat("Churn distribution:\n")
print(table(churn$Churn_yes))
cat("\n")

# ============================================================================
# SECTION 2: TRAIN-TEST SPLIT (IMBALANCED DATA)
# ============================================================================

cat(paste(rep("=", 70), collapse=""), "\n")
cat("SECTION 2: Creating Train-Test Split (70-30)\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

# Split data (70% train, 30% test)
set.seed(42)
split <- sample.split(churn$Churn_yes, SplitRatio = 0.7)
train_set <- subset(churn, split == TRUE)
test_set <- subset(churn, split == FALSE)

cat("Training set size:", nrow(train_set), "rows\n")
cat("Test set size:", nrow(test_set), "rows\n\n")

# Export datasets
cat("Exporting test set...\n")
write.xlsx(test_set, "test.xlsx")
cat("Test set exported to 'test.xlsx'\n\n")

# ============================================================================
# SECTION 3: HANDLE CLASS IMBALANCE WITH UNDERSAMPLING
# ============================================================================

cat(paste(rep("=", 70), collapse=""), "\n")
cat("SECTION 3: Handling Class Imbalance with Undersampling\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

# Check class distribution
cat("Original class distribution:\n")
print(table(churn$Churn_yes))
cat("Proportion:\n")
print(prop.table(table(churn$Churn_yes)))
cat("\n")

# Convert to factor for undersampling
churn$Churn_yes <- as.factor(churn$Churn_yes)

balance_dataset <- function(data, target_col) {
  target_values <- data[[target_col]]
  levels_list <- levels(target_values)
  min_count <- min(table(target_values))
  balanced_list <- lapply(levels_list, function(cls) {
    rows <- data[target_values == cls, , drop = FALSE]
    rows[sample(nrow(rows), min_count), , drop = FALSE]
  })
  balanced <- do.call(rbind, balanced_list)
  rownames(balanced) <- NULL
  balanced
}

# Apply undersampling
cat("Applying undersampling...\n")
set.seed(123)
undersampled <- balance_dataset(churn, "Churn_yes")

cat("\nBalanced class distribution:\n")
print(table(undersampled$Churn_yes))
cat("\n")

# New train-test split on balanced data
cat("Creating train-test split on balanced data...\n")
split_balanced <- sample.split(undersampled$Churn_yes, SplitRatio = 0.7)
train_balanced <- subset(undersampled, split_balanced == TRUE)
test_balanced <- subset(undersampled, split_balanced == FALSE)

cat("Balanced training set size:", nrow(train_balanced), "rows\n")
cat("Balanced test set size:", nrow(test_balanced), "rows\n\n")

# Export balanced datasets
cat("Exporting balanced datasets...\n")
write.xlsx(train_balanced, "train_balanced.xlsx")
write.xlsx(undersampled, "churn_balanced.xlsx")
cat("Balanced training set exported to 'train_balanced.xlsx'\n")
cat("Full balanced dataset exported to 'churn_balanced.xlsx'\n\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat(paste(rep("=", 70), collapse=""), "\n")
cat("DATA CLEANING COMPLETED\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

cat("Files created:\n")
cat("1. test.xlsx - Test set (imbalanced)\n")
cat("2. train_balanced.xlsx - Balanced training set\n")
cat("3. churn_balanced.xlsx - Full balanced dataset\n\n")

cat("Original data:", nrow(churn), "rows\n")
cat("Balanced data:", nrow(undersampled), "rows\n")
cat("Data cleaning completed successfully!\n")