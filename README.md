# footydotpy
Football prediction algorithms
Importing Libraries: The script begins by importing necessary Python libraries such as Pandas for data manipulation, NumPy for numerical operations, Seaborn and Matplotlib for data visualization, and scikit-learn for machine learning tools.

Loading Data: Two CSV files containing football match data for the 2020–2021 and 2021–2022 seasons are loaded using the load_data function. These data files likely contain information about each match, such as team names, scores, and match statistics.

Data Cleaning: The clean_data function is used to preprocess the loaded data. It performs several data cleaning operations, including removing rows with missing values (NaN), stripping whitespace from team names and other string columns, and ensuring consistency in data format.

Feature Selection: The script defines a list of feature columns (feature_columns) that are used as input features for the machine learning model. These features include statistics like goals, shots, and fouls for both home and away teams. The target column (target_column) is the one we want to predict, which is the "Full-Time Result" (FTR) of each match (e.g., 'Home Win', 'Draw', 'Away Win').

Encoding Categorical Columns: Some of the columns in the data are categorical (e.g., team names and half-time results), which need to be converted into numeric form for machine learning. The encode_categorical_columns function converts these categorical columns into categorical codes.

Model Training: The script uses a Random Forest Classifier (RandomForestClassifier from scikit-learn) to build a machine learning model. This model is trained on the training data (X_train and y_train), where X_train contains the input features, and y_train contains the target labels (FTR).

Model Evaluation: After training the model, it is evaluated using the test data (X_test and y_test). The evaluate_model function predicts the match outcomes on the test data and calculates accuracy and a classification report (including precision, recall, and F1-score).

Displaying Results: The script then displays the predicted results alongside the actual results, along with probability rankings for each match outcome (Home Win, Draw, Away Win). It does this for each match in the 2021–2022 season.

Correlation Matrix Visualization: The code includes functions (display_correlation_matrix) to visualize correlation matrices for various columns, including match results and match statistics. These matrices help analyze relationships between different variables.

Main Function: The main function orchestrates all the steps described above. It loads the data, preprocesses it, trains the model, evaluates the model, displays results, and visualizes correlation matrices for both seasons.

Execution: Finally, the script runs the main function when executed as a standalone script.

Data used: https://www.kaggle.com/datasets/saife245/english-premier-league
