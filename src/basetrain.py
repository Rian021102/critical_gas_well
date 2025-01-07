from sklearn.linear_model import LinearRegression,Ridge,Lasso
#import SVR model
from sklearn.svm import SVR
#import RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
#import XGBRegressor model
from xgboost import XGBRegressor
#import mean_squared_error
#import ridge regression model
from sklearn.linear_model import Ridge
# import laso regression model
from sklearn.linear_model import Lasso
#import elastic net regression model
from sklearn.linear_model import ElasticNet
#import MLP regressor model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
#import mean_absolute_error
from sklearn.metrics import mean_absolute_error
# import R2 score
from sklearn.metrics import r2_score


def base_train(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):
    def evaluate_model(model, X_train, y_train, X_test, y_test, use_scaled=False):
        # Decide which data to use based on `use_scaled`
        if use_scaled:
            X_t, y_t = X_train_scaled, X_test_scaled
        else:
            X_t, y_t = X_train, X_test
        model.fit(X_t, y_train)
        y_pred = model.predict(y_t)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_name = type(model).__name__
        print(f"{model_name}:")
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R2 Score:", r2)
    evaluate_model(LinearRegression(), X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(Ridge(),X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(Lasso(),X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(SVR(), X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(ElasticNet(),X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(RandomForestRegressor(),X_train, y_train, X_test, y_test, use_scaled=False)
    evaluate_model(AdaBoostRegressor(),X_train, y_train, X_test, y_test, use_scaled=False)
    evaluate_model(XGBRegressor(),X_train, y_train, X_test, y_test, use_scaled=False)

    return


def mlp_reg(X_train_scaled, y_train, X_test_scaled, y_test):
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000,
                       activation='relu', solver='adam', alpha=0.0001,
                       batch_size='auto', learning_rate='constant',
                       learning_rate_init=0.001, power_t=0.5, max_fun=15000)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MLP Regressor:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)
    return
