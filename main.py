import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset with ordinal encoding."""
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Drop missing values
    df = df.dropna()
    
    # Identify target column (2nd to last column)
    cols = df.columns
    target_column_1 = cols[-2] # time to employment
    target_column_2 = cols[-1] # job title
        
    # Features (X) and target (y)
    X = df.drop([target_column_1, target_column_2], axis=1)
    y1 = df[target_column_1]
    y2 = df[target_column_2]

    # Define ordinal mapping
    ordinal_mapping = {
        "below average (2.75 - 5)": 1,
        "average (1.75 - 2.5)": 2,
        "above average (1.0 - 1.5)": 3,
        "low": 0,
        "moderate": 1,
        "high": 2,
        "yes": 1,
        "no": 0
    }
    
    # Apply ordinal encoding to categorical columns
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].str.lower().map(ordinal_mapping)
    
    # Drop any rows with unmapped categories (if any)
    X = X.dropna()
    y1 = y1.loc[X.index]  # Keep y aligned with X
    y2 = y2.loc[X.index]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y1, y2, scaler

def train_models(X_train, y1_train, y2_train):
    """Train separate models for time to employment (SVR) and job title (SVC)."""
    # Train SVR model for target 1 (Time to employment)
    svr_model = SVR()
    svr_model.fit(X_train, y1_train)

    # Train SVM model for target 2 (Job title classification)
    svm_model = SVC()
    svm_model.fit(X_train, y2_train)
    
    return svr_model, svm_model

def evaluate_models(svr_model, svm_model, X_test, y1_test, y2_test):
    """Evaluate both models and print metrics."""
    # Evaluate SVR model (Time to employment)
    y1_pred = svr_model.predict(X_test)
    mse = mean_squared_error(y1_test, y1_pred)
    r2 = r2_score(y1_test, y1_pred)
    print("SVR Model - MSE:", mse)
    print("SVR Model - R² Score:", r2)

    # Evaluate SVM model (Job title)
    y2_pred = svm_model.predict(X_test)
    accuracy = (y2_pred == y2_test).mean()  # Classification accuracy
    print("SVM Model - Accuracy:", accuracy)

def save_models(svr_model, svm_model, scaler, svr_filename='svr_model.pkl', svm_filename='svm_model.pkl', scaler_filename='scaler.pkl'):
    """Save the trained models and scaler to disk."""
    joblib.dump(svr_model, svr_filename)
    joblib.dump(svm_model, svm_filename)
    joblib.dump(scaler, scaler_filename)

def main():
    # Load and preprocess the data
    X_scaled, y1, y2, scaler = load_and_preprocess_data('dataset.xlsx')
    
    # Train/test split
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X_scaled, y1, y2, test_size=0.2, random_state=42)
    
    # Train the models
    svr_model, svm_model = train_models(X_train, y1_train, y2_train)
    
    # Evaluate the models
    evaluate_models(svr_model, svm_model, X_test, y1_test, y2_test)
    
    # Save the models and scaler
    save_models(svr_model, svm_model, scaler)

if __name__ == '__main__':
    main()
