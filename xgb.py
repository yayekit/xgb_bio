import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from datetime import datetime
import threading

class SensorDataSimulator:
    def __init__(self, num_sensors, num_features):
        self.num_sensors = num_sensors
        self.num_features = num_features
    
    def generate_data(self, num_samples):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = np.random.rand(num_samples, self.num_sensors * self.num_features)
        target = np.sum(data[:, :self.num_sensors], axis=1) + np.random.randn(num_samples) * 0.1
        return pd.DataFrame(data, columns=[f'sensor_{i}_feature_{j}' for i in range(self.num_sensors) for j in range(self.num_features)]), pd.Series(target), timestamp

class XGBoostContinuousLearner:
    def __init__(self, params):
        self.model = None 
        self.params = params
        self.data_buffer = pd.DataFrame()
        self.target_buffer = pd.Series()
        self.buffer_size = 1000
        self.min_samples_retrain = 100
        self.last_train_time = None
    
    def update_data(self, new_data, new_target):
        self.data_buffer = pd.concat([self.data_buffer, new_data]).tail(self.buffer_size)
        self.target_buffer = pd.concat([self.target_buffer, new_target]).tail(self.buffer_size)
    
    def train_model(self):
        if len(self.data_buffer) < self.min_samples_retrain:
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.data_buffer, self.target_buffer, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        if self.model is None:
            self.model = xgb.train(self.params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)
        else:
            self.model = xgb.train(self.params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False, xgb_model=self.model)
        
        y_pred = self.model.predict(dtest)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model trained at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, MSE: {mse:.4f}")
        self.last_train_time = time.time()
    
    def predict(self, data):
        if self.model is None:
            return None
        dmatrix = xgb.DMatrix(data)
        return self.model.predict(dmatrix)

def simulation_loop(sensor_simulator, xgboost_learner):
    while True:
        # Generate new sensor data
        new_data, new_target, timestamp = sensor_simulator.generate_data(10)
        
        # Update the learner's data
        xgboost_learner.update_data(new_data, new_target)
        
        # Train the model if enough new data has been collected
        if xgboost_learner.last_train_time is None or time.time() - xgboost_learner.last_train_time > 60:  # Train every 60 seconds
            xgboost_learner.train_model()
        
        # Make predictions on the new data
        predictions = xgboost_learner.predict(new_data)
        if predictions is not None:
            mse = mean_squared_error(new_target, predictions)
            print(f"Predictions made at {timestamp}, MSE: {mse:.4f}")
        
        time.sleep(1)  # Wait for 1 second before next iteration

if __name__ == "__main__":
    num_sensors = 5
    num_features = 3
    sensor_simulator = SensorDataSimulator(num_sensors, num_features)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    xgboost_learner = XGBoostContinuousLearner(params)
    
    simulation_thread = threading.Thread(target=simulation_loop, args=(sensor_simulator, xgboost_learner))
    simulation_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        