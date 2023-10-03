import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(data):
    data = data.copy()
    data = data.dropna(how='all')
    data['HomeTeam'] = data['HomeTeam'].astype(str).str.strip()
    data['AwayTeam'] = data['AwayTeam'].astype(str).str.strip()
    string_columns = data.select_dtypes(include='object').columns
    for col in string_columns:
        data[col] = data[col].str.strip()
    
    return data


def encode_categorical_columns(data, columns):
    data = data.copy()
    for column in columns:
        data[column] = data[column].astype('category').cat.codes
    return data

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rfc.fit(X_train, y_train)
    return rfc

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return y_pred

def display_results(y_test, y_pred, y_pred_proba, matches):
    for actual, pred, proba, (home, away) in zip(y_test, y_pred, y_pred_proba, matches):
        print(f"Match: {home} vs {away}")
        print(f"Actual Result: {actual}, Predicted Result: {pred}")
        sorted_probs = sorted(zip(['Away Win', 'Draw', 'Home Win'], proba), key=lambda x: x[1], reverse=True)
        print(f"Probability Rankings:")
        for outcome, probability in sorted_probs:
            print(f"{outcome}: {probability*100:.2f}%")
        print("------------------------------------------------------")

def display_correlation_matrix(data, title, columns_to_correlate):
    data = data.copy()
    if 'HTR' in columns_to_correlate:
        data['HTR'] = data['HTR'].astype('category').cat.codes
    if 'FTR' in columns_to_correlate:
        data['FTR'] = data['FTR'].astype('category').cat.codes
    
    available_columns = [col for col in columns_to_correlate if col in data.columns]
    correlation_matrix = data[available_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
    plt.title(title)
    plt.show()
   
def main():
    filepath_2020_2021 = r"C:\data_2020_2021.csv"
    filepath_2021_2022 = r"C:\data_2021_2022.csv"
    
    data_2020_2021 = load_data(filepath_2020_2021)
    data_2021_2022 = load_data(filepath_2021_2022)
    
    data_2020_2021 = clean_data(data_2020_2021)
    data_2021_2022 = clean_data(data_2021_2022)
    
    feature_columns = ['HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 
                       'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    target_column = 'FTR'
    
    X_train = data_2020_2021[feature_columns]
    y_train = data_2020_2021[target_column]
    X_test = data_2021_2022[feature_columns]
    y_test = data_2021_2022[target_column]

    categorical_columns = ['HomeTeam', 'AwayTeam', 'HTR']
    X_train = encode_categorical_columns(X_train, categorical_columns)
    X_test = encode_categorical_columns(X_test, categorical_columns)

    rfc = train_model(X_train, y_train)
    y_pred = evaluate_model(rfc, X_test, y_test)
    y_pred_proba = rfc.predict_proba(X_test)
   
    matches = zip(data_2021_2022['HomeTeam'], data_2021_2022['AwayTeam'])
    display_results(y_test, y_pred, y_pred_proba, matches)
    
    print(f"Total Matches Displayed: {len(X_test)}")

    columns_to_correlate = ['FTR','HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 
                             'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    display_correlation_matrix(data_2020_2021, "Correlation Matrix for 2020-2021 Season", columns_to_correlate)
    display_correlation_matrix(data_2021_2022, "Correlation Matrix for 2021-2022 Season", columns_to_correlate)

if __name__ == "__main__":
    main()
