#import random forest regressor
from xgboost import XGBRegressor
# import mean squared error
from sklearn.metrics import mean_squared_error
# import r2 score
from sklearn.metrics import r2_score
# import mean absolute error
from sklearn.metrics import mean_absolute_error
# import pickle
import pickle
from pathlib import Path

def train_xgb(X_train,y_train,X_test,y_test):
    """Train a Random Forest model and evaluate its performance.

    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values

    Returns:
        tuple: (mse, r2, mae, y_pred, feature_importances)
    """
    #train the model
    model = XGBRegressor()
    model.fit(X_train, y_train)
    #predict the test data
    y_pred = model.predict(X_test)
    #calculate the mean squared error
    #feature importance
    feature_importances = model.feature_importances_
    mse = mean_squared_error(y_test, y_pred)
    #calculate the r2 score
    r2 = r2_score(y_test, y_pred)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    #save model
    model_path = Path('P:/project/pythonpro/myvenv/gas-well-mon/model_Rate')
    model_path.mkdir(parents=True, exist_ok=True)
    with open(model_path / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return mse, r2, mae, y_pred, feature_importances


