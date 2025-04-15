import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset with ordinal encoding."""
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Drop missing values
    df = df.dropna()
    
    # Identify target column (2nd to last column)
    cols = df.columns
    target_column = cols[-1]
        
    # Features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
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
    y = y.loc[X.index]  # Keep y aligned with X
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_svr_model(X_train, y_train):
    """Train the SVR model."""
    model = SVR()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print the metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("MSE:", mse)
    print("R² Score:", r2)

def save_model(model, scaler, model_filename='svr_model.pkl', scaler_filename='scaler.pkl'):
    """Save the trained model and scaler to disk."""
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

def main():
    # Load and preprocess the data
    X_scaled, y, scaler = load_and_preprocess_data('dataset.xlsx')
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train SVR model
    model = train_svr_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the trained model and scaler
    save_model(model, scaler)

if __name__ == '__main__':
    main()
