# Exoplanet Detection using a 1D CNN

## Project Description

This project presents a machine learning model for classifying stars as either having an exoplanet or not. The classification is based on the light flux curve from each star, obtained from the [NASA Kepler Labeled Time Series Data](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data).

The main challenge of this project is handling a highly imbalanced dataset, where the number of stars without exoplanets significantly outnumbers those with them.

## Tech Stack & Key Libraries

  * TensorFlow / Keras: For building and training the neural network model.
  * Pandas / NumPy: For data loading, processing, and manipulation.
  * Scikit-learn: For feature scaling (StandardScaler) and model evaluation (classification_report, confusion_matrix).
  * Imbalanced-learn: For addressing the class imbalance problem using SMOTE (oversampling) and RandomUnderSampler (undersampling) techniques.
  * Matplotlib / Seaborn: For visualizing the training history and model results.

## Methodology

The project pipeline is divided into the following key stages:

1.  Data Preprocessing:

      * Loading the training and test datasets.
      * Handling outliers by clipping values at the 1st and 99th percentiles.
      * Feature scaling using StandardScaler to normalize the data.

2.  Handling Class Imbalance:

      * To combat the imbalance, an imblearn.pipeline.Pipeline was created.
      * First, SMOTE is applied to artificially increase the number of minority class instances (stars with exoplanets).
      * Then, RandomUnderSampler is used to reduce the number of majority class instances.
      * This created a more balanced dataset for training.

3.  Model Architecture:

      * A 1D Convolutional Neural Network (1D CNN) was built, which is well-suited for analyzing time-series data like light curves.
      * The model includes Conv1D, MaxPooling1D, and Dropout layers to prevent overfitting.
      * The ReLU activation function is used in the hidden layers, and sigmoid is used in the output layer for binary classification.

4.  Training and Evaluation:

      * The model is compiled with the Adam optimizer and binary_crossentropy loss function.
      * An EarlyStopping callback is used to automatically stop the training process if performance on the validation set ceases to improve.
      * Results are evaluated using metrics such as accuracy, balanced_accuracy, ROC AUC score, as well as a confusion matrix and a classification report.

## Results

  * The model achieves a high overall accuracy of \~98% on the test set.
  * However, due to the severe imbalance, the metrics for the minority class (stars with exoplanets) are significantly lower. The recall for this class is 40%, meaning the model identifies 2 out of the 3 exoplanets in the test set.
  * The Balanced Accuracy is \~69%, which more objectively reflects the model's performance on the imbalanced data.
