###############################################################################
# Customer Churn Model Training (R)
###############################################################################
# Trains multiple supervised learning models on the cleaned churn datasets:
#   1. Logistic Regression with Stepwise Selection
#   2. Decision Tree (CART)
#   3. Flexible Discriminant Analysis (FDA)
#   4. Random Forest
#
# CLI flags (all optional):
#   --data-dir=PATH            Override automatic data folder discovery
#   --train-file=FILENAME      Specify training dataset filename
#   --test-file=FILENAME       Specify test dataset filename
#   --output-dir=PATH          Where to store derived artifacts (metrics, plots, ...)
#   --metrics-file=NAME.[csv|xlsx]  Metrics report file placed under output-dir
#   --prob-threshold=FLOAT     Logistic regression classification threshold
#   --skip-assumptions         Bypass Box's M and Mardia tests when needed
#
# The script mirrors the preprocessing assumptions from data_cleaning.R and
# reports evaluation metrics plus assumption checks in a structured workflow.
###############################################################################

## ---- Configure Project Library ---------------------------------------------
project_lib <- file.path(getwd(), ".r-lib")
if (!dir.exists(project_lib)) {
  dir.create(project_lib, recursive = TRUE)
}
.libPaths(c(project_lib, .libPaths()))

## ---- Load Required Libraries -------------------------------------------------
required_packages <- c(
  "openxlsx", "MASS", "rpart", "rpart.plot",
  "mda", "randomForest"
)

missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  install.packages(
    missing_packages,
    repos = "https://cloud.r-project.org",
    dependencies = TRUE,
    lib = project_lib,
    type = "binary"
  )
}
invisible(lapply(required_packages, require, character.only = TRUE))

## ---- Helper Utilities --------------------------------------------------------
separator_line <- function(char = "=", width = 70) {
  paste(rep(char, width), collapse = "")
}

print_section <- function(title) {
  cat("\n", separator_line(), "\n", title, "\n", separator_line(), "\n\n", sep = "")
}

find_data_dir <- function() {
  wd <- normalizePath(getwd(), winslash = "\\")
  candidates <- unique(c(
    wd,
    file.path(wd, ".."),
    file.path(wd, "../..")
  ))

  for (candidate in candidates) {
    normalized <- normalizePath(candidate, winslash = "\\", mustWork = FALSE)
    data_dir <- file.path(normalized, "data")
    if (dir.exists(data_dir)) {
      return(data_dir)
    }
  }
  stop("Unable to locate the data directory. Please update the path resolution logic.")
}

resolve_data_file <- function(data_dir, filename_options) {
  for (candidate in filename_options) {
    candidate_path <- file.path(data_dir, candidate)
    if (file.exists(candidate_path)) {
      return(candidate_path)
    }
  }
  stop("Could not find any of the expected files: ", paste(filename_options, collapse = ", "))
}

coerce_categoricals <- function(df, categorical_cols) {
  available_cols <- intersect(categorical_cols, names(df))
  for (col in available_cols) {
    df[[col]] <- as.factor(df[[col]])
  }
  character_cols <- names(df)[vapply(df, is.character, logical(1))]
  for (col in setdiff(character_cols, available_cols)) {
    df[[col]] <- as.factor(df[[col]])
  }
  df
}

prepare_datasets <- function(train_df, test_df, categorical_cols, target_col) {
  train_df <- coerce_categoricals(train_df, categorical_cols)
  test_df <- coerce_categoricals(test_df, categorical_cols)

  if (!target_col %in% names(train_df)) {
    stop("Target column '", target_col, "' not found in training data.")
  }
  if (!target_col %in% names(test_df)) {
    stop("Target column '", target_col, "' not found in test data.")
  }

  target_levels <- sort(unique(train_df[[target_col]]))
  if (length(target_levels) != 2) {
    stop("Target variable must be binary. Found levels: ", paste(target_levels, collapse = ", "))
  }

  train_df[[target_col]] <- factor(train_df[[target_col]], levels = target_levels)
  test_df[[target_col]] <- factor(test_df[[target_col]], levels = target_levels)

  list(
    train = train_df,
    test = test_df,
    positive_level = tail(target_levels, 1),
    negative_level = head(target_levels, 1)
  )
}

safe_divide <- function(num, denom) {
  if (is.na(num) || is.na(denom) || denom == 0) {
    return(NA_real_)
  }
  num / denom
}

compute_confusion_stats <- function(truth, preds, positive_level, negative_level) {
  truth <- factor(truth, levels = c(negative_level, positive_level))
  preds <- factor(preds, levels = c(negative_level, positive_level))
  cm <- table(Prediction = preds, Reference = truth)

  tp <- sum(preds == positive_level & truth == positive_level)
  tn <- sum(preds == negative_level & truth == negative_level)
  fp <- sum(preds == positive_level & truth == negative_level)
  fn <- sum(preds == negative_level & truth == positive_level)
  total <- tp + tn + fp + fn

  accuracy <- safe_divide(tp + tn, total)
  precision <- safe_divide(tp, tp + fp)
  recall <- safe_divide(tp, tp + fn)
  specificity <- safe_divide(tn, tn + fp)
  f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
    NA_real_
  } else {
    2 * precision * recall / (precision + recall)
  }
  balanced_accuracy <- mean(c(recall, specificity), na.rm = TRUE)
  pe <- safe_divide((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total^2)
  kappa <- if (is.na(pe) || pe == 1) NA_real_ else (accuracy - pe) / (1 - pe)

  list(
    table = cm,
    overall = c(Accuracy = accuracy, Kappa = kappa),
    byClass = c(
      Sensitivity = recall,
      Specificity = specificity,
      Precision = precision,
      Recall = recall,
      F1 = f1,
      `Balanced Accuracy` = balanced_accuracy
    )
  )
}

evaluate_model <- function(name, truth, preds, positive_level, negative_level) {
  cat(sprintf("%s - Confusion Matrix:\n", name))
  metrics <- compute_confusion_stats(truth, preds, positive_level, negative_level)
  print(metrics$table)
  cat("\nPerformance summary (rounded):\n")
  summary_vec <- round(c(metrics$overall, metrics$byClass), 3)
  print(summary_vec)
  cat("\n")
  metrics
}

numeric_predictors <- function(df, target_col) {
  predictors <- df[, setdiff(names(df), target_col), drop = FALSE]
  numeric_cols <- predictors[, vapply(predictors, is.numeric, logical(1)), drop = FALSE]
  numeric_cols
}

run_assumption_tests <- function(df, target_col) {
  numerics <- numeric_predictors(df, target_col)
  if (ncol(numerics) == 0) {
    cat("Skipping assumption tests: no numeric predictors available.\n\n")
    return(invisible(NULL))
  }

  if (ncol(numerics) >= 2) {
    if (requireNamespace("biotools", quietly = TRUE)) {
      cat("Box's M Test (Homogeneity of Covariance):\n")
      boxm_out <- biotools::boxM(numerics, df[[target_col]])
      print(boxm_out)
      cat("\n")
    } else {
      cat("Box's M Test skipped: install.packages('biotools') to enable this diagnostic.\n\n")
    }
  } else {
    cat("Box's M Test skipped: requires at least two numeric predictors.\n\n")
  }

  sample_size <- min(500, nrow(numerics))
  set.seed(42)
  sample_rows <- sample(seq_len(nrow(numerics)), sample_size)
  sample_data <- numerics[sample_rows, , drop = FALSE]
  if (ncol(sample_data) > 1) {
    if (requireNamespace("MVN", quietly = TRUE)) {
      cat("Mardia's Test for Multivariate Normality:\n")
      mardia_out <- MVN::mvn(sample_data, mvnTest = "mardia", multivariatePlot = "qq")
      print(mardia_out$multivariateNormality)
      cat("\n")
    } else {
      cat("Mardia's Test skipped: install.packages('MVN') to enable this diagnostic.\n\n")
    }
  } else {
    cat("Mardia's Test skipped: requires more than one numeric predictor.\n\n")
  }
}

parse_cli_args <- function(defaults) {
  # Minimal CLI parser that accepts --key=value pairs or standalone flags.
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- defaults

  parse_arg <- function(arg) {
    stripped <- sub("^--", "", arg)
    if (!grepl("=", stripped, fixed = TRUE)) {
      return(list(key = stripped, value = NA_character_))
    }
    parts <- strsplit(stripped, "=", fixed = TRUE)[[1]]
    list(key = parts[1], value = paste(parts[-1], collapse = "="))
  }

  for (arg in args) {
    if (!startsWith(arg, "--")) {
      next
    }
    kv <- parse_arg(arg)
    key <- kv$key
    value <- kv$value
    switch(
      key,
      "data-dir" = {
        cfg$data_dir <- normalizePath(value, winslash = "\\", mustWork = FALSE)
      },
      "train-file" = {
        cfg$train_file <- value
      },
      "test-file" = {
        cfg$test_file <- value
      },
      "output-dir" = {
        cfg$output_dir <- normalizePath(value, winslash = "\\", mustWork = FALSE)
      },
      "prob-threshold" = {
        cfg$prob_threshold <- as.numeric(value)
      },
      "metrics-file" = {
        cfg$metrics_file <- value
      },
      "skip-assumptions" = {
        cfg$skip_assumptions <- TRUE
      },
      {
        warning(sprintf("Unknown CLI argument ignored: --%s", key))
      }
    )
  }

  if (is.na(cfg$prob_threshold) || cfg$prob_threshold <= 0 || cfg$prob_threshold >= 1) {
    cfg$prob_threshold <- defaults$prob_threshold
    warning("Probability threshold must be in (0,1); defaulting to 0.5")
  }
  cfg
}

print_config <- function(cfg) {
  # Surface the active configuration so runs remain traceable in logs.
  cat("Configuration:\n")
  cat(sprintf("  Data directory: %s\n", cfg$data_dir))
  cat(sprintf(
    "  Training file preference: %s\n",
    ifelse(is.na(cfg$train_file), "auto", cfg$train_file)
  ))
  cat(sprintf(
    "  Test file preference: %s\n",
    ifelse(is.na(cfg$test_file), "auto", cfg$test_file)
  ))
  cat(sprintf("  Output directory: %s\n", cfg$output_dir))
  cat(sprintf("  Metrics file: %s\n", cfg$metrics_file))
  cat(sprintf("  Probability threshold: %.2f\n", cfg$prob_threshold))
  cat(sprintf("  Skip assumption tests: %s\n\n", ifelse(cfg$skip_assumptions, "yes", "no")))
}

safe_metric <- function(vec, name) {
  if (is.null(vec) || is.na(name) || !name %in% names(vec)) {
    return(NA_real_)
  }
  unname(vec[[name]])
}

summarize_metrics <- function(metrics_list) {
  rows <- lapply(names(metrics_list), function(model_name) {
    metric <- metrics_list[[model_name]]
    data.frame(
      Model = model_name,
      Accuracy = safe_metric(metric$overall, "Accuracy"),
      Kappa = safe_metric(metric$overall, "Kappa"),
      Sensitivity = safe_metric(metric$byClass, "Sensitivity"),
      Specificity = safe_metric(metric$byClass, "Specificity"),
      Precision = safe_metric(metric$byClass, "Precision"),
      Recall = safe_metric(metric$byClass, "Recall"),
      F1 = safe_metric(metric$byClass, "F1"),
      BalancedAccuracy = safe_metric(metric$byClass, "Balanced Accuracy"),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

persist_metrics <- function(summary_df, metrics_list, metrics_path) {
  dir.create(dirname(metrics_path), recursive = TRUE, showWarnings = FALSE)
  extension <- tolower(tools::file_ext(metrics_path))
  if (identical(extension, "csv") || extension == "") {
    write.csv(summary_df, metrics_path, row.names = FALSE)
    return(invisible(metrics_path))
  }

  wb <- createWorkbook()
  addWorksheet(wb, "summary")
  writeData(wb, "summary", summary_df)

  for (model_name in names(metrics_list)) {
    sheet_name <- substr(model_name, 1, 31)
    addWorksheet(wb, sheet_name)
    cm <- as.data.frame.matrix(metrics_list[[model_name]]$table)
    cm <- cbind(Predicted = rownames(cm), cm, row.names = NULL)
    writeData(wb, sheet_name, cm)
  }

  saveWorkbook(wb, metrics_path, overwrite = TRUE)
  invisible(metrics_path)
}

## ---- Path Resolution & Data Loading -----------------------------------------
cat("Initializing churn model training pipeline...\n")
default_data_dir <- find_data_dir()
default_config <- list(
  data_dir = default_data_dir,
  train_file = NA_character_,
  test_file = NA_character_,
  output_dir = file.path(default_data_dir, "model_outputs"),
  metrics_file = "model_metrics_summary.xlsx",
  prob_threshold = 0.5,
  skip_assumptions = FALSE
)
config <- parse_cli_args(default_config)

if (!dir.exists(config$output_dir)) {
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)
}

print_config(config)

train_candidates <- if (!is.na(config$train_file)) {
  c(config$train_file)
} else {
  c("train_balanced_smote.xlsx", "train_balanced.xlsx")
}
test_candidates <- if (!is.na(config$test_file)) {
  c(config$test_file)
} else {
  c("test.xlsx")
}

train_path <- resolve_data_file(config$data_dir, train_candidates)
test_path <- resolve_data_file(config$data_dir, test_candidates)

cat(sprintf("Training data source: %s\n", basename(train_path)))
cat(sprintf("Test data source: %s\n\n", basename(test_path)))

raw_train <- read.xlsx(train_path)
raw_test <- read.xlsx(test_path)
cat(sprintf("Training set -> %s rows x %s cols\n", nrow(raw_train), ncol(raw_train)))
cat(sprintf("Test set -> %s rows x %s cols\n\n", nrow(raw_test), ncol(raw_test)))

categorical_cols <- c(
  "Product.Category", "Payment.Method", "Returns",
  "Customer.Name", "Gender"
)
identifier_cols <- c("Customer.Name")

target_column <- "Churn_yes"
datasets <- prepare_datasets(raw_train, raw_test, categorical_cols, target_column)
train_df <- datasets$train
test_df <- datasets$test
positive_level <- datasets$positive_level
negative_level <- datasets$negative_level

drop_identifier_columns <- function(df, cols) {
  drop_cols <- intersect(cols, names(df))
  if (length(drop_cols) > 0) {
    df[drop_cols] <- NULL
  }
  df
}

train_df <- drop_identifier_columns(train_df, identifier_cols)
test_df <- drop_identifier_columns(test_df, identifier_cols)

cat(
  "Identifier-like columns removed from modeling set:",
  ifelse(length(identifier_cols) == 0, "none", paste(identifier_cols, collapse = ", ")),
  "\n"
)
cat("Remaining predictors: ", paste(setdiff(names(train_df), target_column), collapse = ", "), "\n\n", sep = "")

cat("Data preparation completed using factors consistent with data_cleaning.R.\n\n")

## ---- Section 1: Logistic Regression ----------------------------------------
print_section("SECTION 1: Logistic Regression with Stepwise Selection")

cat("Fitting full logistic regression model...\n")
logistic_full <- glm(
  reformulate(termlabels = setdiff(names(train_df), target_column), response = target_column),
  family = binomial(link = "logit"),
  data = train_df
)

cat("\nModel Summary:\n")
print(summary(logistic_full))
cat("\nANOVA Table:\n")
print(anova(logistic_full, test = "Chisq"))

cat("\nPerforming stepwise variable selection...\n")
stepwise_logistic <- stepAIC(logistic_full, direction = "both", trace = FALSE)
cat("Selected model AIC:", AIC(stepwise_logistic), "\n\n")
print(summary(stepwise_logistic))

cat("\nOdds Ratios:\n")
odds_ratios <- exp(coef(stepwise_logistic))
print(odds_ratios)
cat("\n")

logistic_probs <- predict(stepwise_logistic, newdata = test_df, type = "response")
logistic_preds <- ifelse(logistic_probs >= config$prob_threshold, positive_level, negative_level)
logistic_preds <- factor(logistic_preds, levels = levels(test_df[[target_column]]))

model_metrics <- list()
model_metrics$logistic <- evaluate_model(
  "Logistic Regression",
  truth = test_df[[target_column]],
  preds = logistic_preds,
  positive_level = positive_level,
  negative_level = negative_level
)

## ---- Section 2: Decision Tree (CART) ---------------------------------------
print_section("SECTION 2: Decision Tree (CART)")

set.seed(42)
tree_model <- rpart(
  reformulate(termlabels = setdiff(names(train_df), target_column), response = target_column),
  data = train_df,
  method = "class"
)

cat("Decision Tree Summary:\n")
print(tree_model)
cat("\nGenerating tree plot...\n")
rpart.plot(tree_model)

cat("\nFeature Importance (Descending):\n")
tree_importance <- tree_model$variable.importance
if (is.null(tree_importance)) {
  cat("  (importance values unavailable from rpart)\n\n")
} else {
  tree_importance_df <- data.frame(
    Feature = names(tree_importance),
    Overall = as.numeric(tree_importance),
    row.names = NULL
  )
  tree_importance_df <- tree_importance_df[order(-tree_importance_df$Overall), ]
  print(tree_importance_df)
  cat("\n")
}

tree_preds <- predict(tree_model, newdata = test_df, type = "class")
model_metrics$tree <- evaluate_model(
  "Decision Tree",
  truth = test_df[[target_column]],
  preds = tree_preds,
  positive_level = positive_level,
  negative_level = negative_level
)

## ---- Section 3: Assumption Tests -------------------------------------------
print_section("SECTION 3: Statistical Assumption Tests")
if (config$skip_assumptions) {
  cat("Assumption tests skipped via --skip-assumptions flag.\n\n")
} else {
  run_assumption_tests(train_df, target_column)
}

## ---- Section 4: Flexible Discriminant Analysis -----------------------------
print_section("SECTION 4: Flexible Discriminant Analysis (FDA)")

fda_model <- fda(
  reformulate(termlabels = setdiff(names(train_df), target_column), response = target_column),
  data = train_df
)
cat("FDA Model Summary:\n")
print(summary(fda_model))
cat("\n")

fda_preds <- predict(fda_model, newdata = test_df)
model_metrics$fda <- evaluate_model(
  "Flexible Discriminant Analysis",
  truth = test_df[[target_column]],
  preds = fda_preds,
  positive_level = positive_level,
  negative_level = negative_level
)

## ---- Section 5: Random Forest ---------------------------------------------
print_section("SECTION 5: Random Forest")

set.seed(42)
rf_model <- randomForest(
  reformulate(termlabels = setdiff(names(train_df), target_column), response = target_column),
  data = train_df,
  ntree = 500,
  mtry = min(5, floor(sqrt(ncol(train_df) - 1)))
)

cat("Random Forest Summary:\n")
print(rf_model)
cat("\n")

rf_preds <- predict(rf_model, newdata = test_df)
model_metrics$random_forest <- evaluate_model(
  "Random Forest",
  truth = test_df[[target_column]],
  preds = rf_preds,
  positive_level = positive_level,
  negative_level = negative_level
)

cat("Variable Importance Plot generated below.\n")
varImpPlot(rf_model)

metrics_summary <- summarize_metrics(model_metrics)
metrics_output_path <- file.path(config$output_dir, config$metrics_file)
persist_metrics(metrics_summary, model_metrics, metrics_output_path)
cat(sprintf("Metrics summary saved to %s\n\n", metrics_output_path))

## ---- Summary ----------------------------------------------------------------
print_section("MODEL TRAINING COMPLETED")

cat("Models trained and evaluated: Logistic Regression, Decision Tree, FDA, Random Forest.\n")
cat("For detailed metrics, review the confusion matrices printed above.\n")
best_row <- metrics_summary[order(-metrics_summary$Accuracy), ][1, , drop = FALSE]
cat(
  sprintf(
    "Highest accuracy model: %s (%.3f)\n",
    best_row$Model,
    best_row$Accuracy
  )
)
cat(sprintf("Probability threshold applied for Logistic Regression: %.2f\n", config$prob_threshold))

cat("\nPipeline finished successfully.\n")