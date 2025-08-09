# Advanced Machine Learning Models for Wave Energy Optimization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedWECPredictor:
    """
    Advanced machine learning models for WEC power prediction
    Includes ensemble methods, deep learning, and Gaussian processes
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.feature_importance = None
        
    def prepare_advanced_features(self, coordinates, n_wecs):
        """Create advanced spatial and geometric features"""
        positions = coordinates.reshape(n_wecs, 2)
        
        # Basic spatial features
        from scipy.spatial.distance import pdist
        from scipy.spatial import ConvexHull
        
        distances = pdist(positions, 'euclidean')
        
        features = {
            # Distance statistics
            'mean_distance': np.mean(distances) if len(distances) > 0 else 0,
            'std_distance': np.std(distances) if len(distances) > 0 else 0,
            'min_distance': np.min(distances) if len(distances) > 0 else 0,
            'max_distance': np.max(distances) if len(distances) > 0 else 0,
            'median_distance': np.median(distances) if len(distances) > 0 else 0,
            
            # Geometric features
            'x_range': np.max(positions[:, 0]) - np.min(positions[:, 0]),
            'y_range': np.max(positions[:, 1]) - np.min(positions[:, 1]),
            'x_std': np.std(positions[:, 0]),
            'y_std': np.std(positions[:, 1]),
            'centroid_x': np.mean(positions[:, 0]),
            'centroid_y': np.mean(positions[:, 1]),
            
            # Density features
            'wec_count': n_wecs,
        }
        
        # Convex hull features
        try:
            if len(positions) >= 3:
                hull = ConvexHull(positions)
                features['hull_area'] = hull.volume
                features['hull_perimeter'] = hull.area
                features['density'] = n_wecs / hull.volume if hull.volume > 0 else 0
                features['compactness'] = (4 * np.pi * hull.volume) / (hull.area ** 2) if hull.area > 0 else 0
            else:
                features['hull_area'] = 0
                features['hull_perimeter'] = 0
                features['density'] = 0
                features['compactness'] = 0
        except:
            features['hull_area'] = features['x_range'] * features['y_range']
            features['hull_perimeter'] = 2 * (features['x_range'] + features['y_range'])
            features['density'] = n_wecs / features['hull_area'] if features['hull_area'] > 0 else 0
            features['compactness'] = 0
        
        # Spacing analysis
        if len(distances) > 0:
            # Nearest neighbor distances for each WEC
            dist_matrix = np.zeros((n_wecs, n_wecs))
            k = 0
            for i in range(n_wecs):
                for j in range(i+1, n_wecs):
                    dist_matrix[i, j] = distances[k]
                    dist_matrix[j, i] = distances[k]
                    k += 1
            
            nearest_distances = []
            for i in range(n_wecs):
                row = dist_matrix[i, :]
                row = row[row > 0]  # Remove self-distance
                if len(row) > 0:
                    nearest_distances.append(np.min(row))
            
            if nearest_distances:
                features['avg_nearest_distance'] = np.mean(nearest_distances)
                features['std_nearest_distance'] = np.std(nearest_distances)
            else:
                features['avg_nearest_distance'] = 0
                features['std_nearest_distance'] = 0
        else:
            features['avg_nearest_distance'] = 0
            features['std_nearest_distance'] = 0
        
        # Regularity features
        features['position_entropy'] = self._calculate_position_entropy(positions)
        features['clustering_coefficient'] = self._calculate_clustering(positions)
        
        return list(features.values()), list(features.keys())
    
    def _calculate_position_entropy(self, positions):
        """Calculate spatial entropy of WEC positions"""
        try:
            # Divide space into grid and calculate entropy
            x_bins = np.linspace(np.min(positions[:, 0]), np.max(positions[:, 0]), 10)
            y_bins = np.linspace(np.min(positions[:, 1]), np.max(positions[:, 1]), 10)
            
            hist, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_bins, y_bins])
            hist = hist + 1e-10  # Avoid log(0)
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            return entropy
        except:
            return 0
    
    def _calculate_clustering(self, positions):
        """Calculate clustering coefficient"""
        try:
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(positions))
            n = len(positions)
            
            # Define "neighbors" as WECs within certain distance
            threshold = np.mean(dist_matrix) * 0.8
            adj_matrix = (dist_matrix < threshold) & (dist_matrix > 0)
            
            clustering_coeffs = []
            for i in range(n):
                neighbors = np.where(adj_matrix[i])[0]
                if len(neighbors) < 2:
                    clustering_coeffs.append(0)
                    continue
                
                # Count connections between neighbors
                connections = 0
                possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
                
                for j in range(len(neighbors)):
                    for k in range(j+1, len(neighbors)):
                        if adj_matrix[neighbors[j], neighbors[k]]:
                            connections += 1
                
                clustering_coeffs.append(connections / possible_connections if possible_connections > 0 else 0)
            
            return np.mean(clustering_coeffs)
        except:
            return 0
    
    def create_ensemble_models(self):
        """Create advanced ensemble models"""
        self.models = {
            # Gradient Boosting
            'xgboost': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            
            # Advanced Random Forest
            'advanced_rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            
            # Gradient Boosting
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            
            # Neural Network
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            
            # Gaussian Process
            'gaussian_process': GaussianProcessRegressor(
                kernel=Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
        }
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            if model_name in ['neural_network', 'gaussian_process']:
                self.scalers[model_name] = StandardScaler()
            else:
                self.scalers[model_name] = RobustScaler()
    
    def train_models(self, X, y, test_size=0.2):
        """Train all models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Scale features if needed
            scaler = self.scalers[name]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            model_scores[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'model': model,
                'scaler': scaler
            }
            
            print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        # Select best model based on R²
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['r2'])
        self.best_model = model_scores[best_model_name]['model']
        self.best_scaler = model_scores[best_model_name]['scaler']
        
        print(f"\nBest model: {best_model_name}")
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        
        return model_scores
    
    def predict(self, coordinates, n_wecs):
        """Predict power output using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        features, _ = self.prepare_advanced_features(coordinates, n_wecs)
        features_scaled = self.best_scaler.transform([features])
        
        return self.best_model.predict(features_scaled)[0]
    
    def predict_with_uncertainty(self, coordinates, n_wecs):
        """Predict with uncertainty estimation (for Gaussian Process)"""
        if 'gaussian_process' not in self.models:
            return self.predict(coordinates, n_wecs), 0
        
        features, _ = self.prepare_advanced_features(coordinates, n_wecs)
        features_scaled = self.scalers['gaussian_process'].transform([features])
        
        gp_model = self.models['gaussian_process']
        mean, std = gp_model.predict(features_scaled, return_std=True)
        
        return mean[0], std[0]
    
    def save_models(self, filepath_prefix):
        """Save trained models to disk"""
        for name, model in self.models.items():
            joblib.dump(model, f"{filepath_prefix}_{name}.joblib")
            joblib.dump(self.scalers[name], f"{filepath_prefix}_{name}_scaler.joblib")
    
    def load_models(self, filepath_prefix, model_names=None):
        """Load trained models from disk"""
        if model_names is None:
            model_names = ['xgboost', 'lightgbm', 'advanced_rf', 'neural_network']
        
        for name in model_names:
            try:
                self.models[name] = joblib.load(f"{filepath_prefix}_{name}.joblib")
                self.scalers[name] = joblib.load(f"{filepath_prefix}_{name}_scaler.joblib")
            except FileNotFoundError:
                print(f"Model {name} not found, skipping...")

class EnsemblePredictor:
    """Meta-ensemble that combines multiple models"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict(self, coordinates, n_wecs):
        """Ensemble prediction using weighted average"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(coordinates, n_wecs)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, weights=self.weights)
        return ensemble_pred
    
    def predict_with_variance(self, coordinates, n_wecs):
        """Prediction with ensemble variance"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(coordinates, n_wecs)
            predictions.append(pred)
        
        ensemble_mean = np.average(predictions, weights=self.weights)
        ensemble_var = np.average((predictions - ensemble_mean)**2, weights=self.weights)
        
        return ensemble_mean, np.sqrt(ensemble_var)

# Example usage and testing
if __name__ == "__main__":
    # Create and test advanced models
    predictor = AdvancedWECPredictor()
    predictor.create_ensemble_models()
    
    # Generate sample training data
    np.random.seed(42)
    n_samples = 1000
    n_wecs = 49
    
    X_samples = []
    y_samples = []
    
    for _ in range(n_samples):
        # Random coordinates
        coords = np.random.uniform(0, 1000, n_wecs * 2)
        features, feature_names = predictor.prepare_advanced_features(coords, n_wecs)
        
        # Simulate power (simplified)
        base_power = n_wecs * 45000
        distance_factor = 1 + 0.002 * (features[0] - 100)  # mean_distance
        density_factor = 1 - 0.000001 * features[12]  # density
        power = base_power * distance_factor * density_factor + np.random.normal(0, 50000)
        
        X_samples.append(features)
        y_samples.append(max(power, 0))
    
    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples)
    
    print(f"Training data shape: {X_samples.shape}")
    print(f"Feature names: {feature_names}")
    
    # Train models
    model_scores = predictor.train_models(X_samples, y_samples)
    
    # Test prediction
    test_coords = np.random.uniform(0, 1000, n_wecs * 2)
    prediction = predictor.predict(test_coords, n_wecs)
    print(f"\nTest prediction: {prediction/1e6:.2f} MW")
    
    # Test uncertainty prediction
    if 'gaussian_process' in predictor.models:
        mean, std = predictor.predict_with_uncertainty(test_coords, n_wecs)
        print(f"Prediction with uncertainty: {mean/1e6:.2f} ± {std/1e6:.2f} MW")
