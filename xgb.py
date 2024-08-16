import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from datetime import datetime
import threading
from typing import Tuple, Optional

class SensorDataSimulator:
    """
    Simulates sensor data for machine learning tasks.
    
    This class generates synthetic sensor data with a specified number of sensors and features.
    The generated data can be used to simulate real-world sensor readings for testing and
    development of machine learning models.
    """

    def __init__(self, num_sensors: int, num_features: int):
        """
        Initialize the SensorDataSimulator.

        Args:
            num_sensors (int): The number of sensors to simulate.
            num_features (int): The number of features per sensor.
        """
        self.num_sensors = num_sensors
        self.num_features = num_features
    
    def generate_data(self, num_samples: int) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Generate simulated sensor data.

        This method creates a dataset of simulated sensor readings along with a target variable.
        The target is calculated as a sum of the first 'num_sensors' features plus some random noise.

        Args:
            num_samples (int): The number of data points to generate.

        Returns:
            Tuple[pd.DataFrame, pd.Series, str]: A tuple containing:
                - DataFrame of simulated sensor data
                - Series of target values
                - Timestamp string when the data was generated
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate random data for all sensors and features
        data = np.random.rand(num_samples, self.num_sensors * self.num_features)
        
        # Calculate target as sum of first 'num_sensors' columns plus noise
        target = np.sum(data[:, :self.num_sensors], axis=1) + np.random.randn(num_samples) * 0.1
        
        # Create column names for the DataFrame
        columns = [f'sensor_{i}_feature_{j}' for i in range(self.num_sensors) for j in range(self.num_features)]
        
        return pd.DataFrame(data, columns=columns), pd.Series(target), timestamp

class XGBoostContinuousLearner:
    """
    A continuous learning model using XGBoost.

    This class implements a machine learning model that can be continuously updated
    with new data. It uses XGBoost as the underlying algorithm and maintains a buffer
    of recent data for retraining.
    """

    def __init__(self, params: dict, buffer_size: int = 1000, min_samples_retrain: int = 100):
        """
        Initialize the XGBoostContinuousLearner.

        Args:
            params (dict): Parameters for the XGBoost model.
            buffer_size (int): Maximum number of samples to keep in the buffer.
            min_samples_retrain (int): Minimum number of samples required for retraining.
        """
        self.model: Optional[xgb.Booster] = None 
        self.params = params
        self.data_buffer = pd.DataFrame()
        self.target_buffer = pd.Series()
        self.buffer_size = buffer_size
        self.min_samples_retrain = min_samples_retrain
        self.last_train_time: Optional[float] = None
    
    def update_data(self, new_data: pd.DataFrame, new_target: pd.Series) -> None:
        """
        Update the data buffer with new samples.

        This method adds new data to the buffer and removes old data if the buffer size is exceeded.

        Args:
            new_data (pd.DataFrame): New feature data to add to the buffer.
            new_target (pd.Series): New target data to add to the buffer.
        """
        self.data_buffer = pd.concat([self.data_buffer, new_data]).tail(self.buffer_size)
        self.target_buffer = pd.concat([self.target_buffer, new_target]).tail(self.buffer_size)
    
    def train_model(self) -> None:
        """
        Train or update the XGBoost model.

        This method trains a new model or updates the existing model using the current data in the buffer.
        It only trains if there are enough samples in the buffer.
        """
        if len(self.data_buffer) < self.min_samples_retrain:
            return
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data_buffer, self.target_buffer, test_size=0.2, random_state=42)
        
        # Convert data to DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train the model
        self.model = xgb.train(
            self.params, 
            dtrain, 
            num_boost_round=100, 
            evals=[(dtest, 'eval')], 
            early_stopping_rounds=10, 
            verbose_eval=False,
            xgb_model=self.model  # Use the existing model as a starting point if available
        )
        
        # Evaluate the model
        y_pred = self.model.predict(dtest)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model trained at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, MSE: {mse:.4f}")
        
        self.last_train_time = time.time()
    
    def predict(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make predictions using the trained model.

        Args:
            data (pd.DataFrame): The input data for making predictions.

        Returns:
            Optional[np.ndarray]: Predicted values, or None if the model hasn't been trained yet.
        """
        if self.model is None:
            return None
        dmatrix = xgb.DMatrix(data)
        return self.model.predict(dmatrix)

class ContinuousLearningSimulator:
    """
    Simulates a continuous learning environment.

    This class orchestrates the continuous learning process by generating new data,
    updating the model, and making predictions in a loop.
    """

    def __init__(self, sensor_simulator: SensorDataSimulator, xgboost_learner: XGBoostContinuousLearner, train_interval: int = 60):
        """
        Initialize the ContinuousLearningSimulator.

        Args:
            sensor_simulator (SensorDataSimulator): The data generation simulator.
            xgboost_learner (XGBoostContinuousLearner): The continuous learning model.
            train_interval (int): Time interval (in seconds) between model training sessions.
        """
        self.sensor_simulator = sensor_simulator
        self.xgboost_learner = xgboost_learner
        self.train_interval = train_interval
        self.is_running = False

    def simulation_loop(self):
        """
        The main simulation loop.

        This method continuously generates new data, updates the model,
        and makes predictions until the simulation is stopped.
        """
        while self.is_running:
            # Generate new sensor data
            new_data, new_target, timestamp = self.sensor_simulator.generate_data(10)
            
            # Update the learner's data buffer
            self.xgboost_learner.update_data(new_data, new_target)
            
            # Train the model if it's time to do so
            if self.xgboost_learner.last_train_time is None or time.time() - self.xgboost_learner.last_train_time > self.train_interval:
                self.xgboost_learner.train_model()
            
            # Make predictions on the new data
            predictions = self.xgboost_learner.predict(new_data)
            if predictions is not None:
                mse = mean_squared_error(new_target, predictions)
                print(f"Predictions made at {timestamp}, MSE: {mse:.4f}")
            
            # Wait for a short time before the next iteration
            time.sleep(1)

    def start(self):
        """
        Start the simulation in a separate thread.
        """
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.start()

    def stop(self):
        """
        Stop the simulation and wait for the thread to finish.
        """
        self.is_running = False
        if hasattr(self, 'simulation_thread'):
            self.simulation_thread.join()

def main():
    """
    Main function to set up and run the continuous learning simulation.
    """
    # Set up the sensor simulator
    num_sensors = 5
    num_features = 3
    sensor_simulator = SensorDataSimulator(num_sensors, num_features)
    
    # Set up the XGBoost learner with specified parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    xgboost_learner = XGBoostContinuousLearner(params)
    
    # Create and start the simulation
    simulator = ContinuousLearningSimulator(sensor_simulator, xgboost_learner)
    
    try:
        simulator.start()
        print("Simulation started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        simulator.stop()

if __name__ == "__main__":
    main()