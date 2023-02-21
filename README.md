# CategoryClassification
Classification of Product Category from Product Title

This is a Python code that aims to classify products based on their category when given the name of the product. The code imports several packages such as numpy, pandas, nltk, gensim, sklearn, and simpletransformers.

The data preprocessing phase includes reading a CSV file, selecting specific columns, appending additional data, handling null values, converting categorical data to numerical data using label encoding, and dropping duplicates. The product names are then cleaned using the "clean_txt" function.

The word embedding process uses Google's pre-trained BERT model and Word2Vec to convert the cleaned product names into vectors. After obtaining the vectorized representation of the product names, the data is split into training and test sets, and an SVM classification model is used for modeling.

The code then performs hyperparameter tuning using grid search to find the best values for the hyperparameters of the SVM model. Finally, the code reports the classification metrics such as accuracy, precision, recall, and f1-score for both the regular SVM model and the tuned SVM model.

The results of the models are compared, and the best-performing model is identified. The code prints out the classification report for both models and saves the scores in a Pandas dataframe for comparison.
