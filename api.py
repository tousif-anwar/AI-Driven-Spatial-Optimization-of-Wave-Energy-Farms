"""
REST API for Wave Energy Farm Optimization
Provides endpoints for optimization, prediction, and analysis
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from economic_model import WECEconomicModel
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global models (would be loaded from saved models in production)
economic_model = WECEconomicModel()

class WECOptimizationAPI:
    """API wrapper for WEC optimization services"""
    
    def __init__(self):
        self.economic_model = WECEconomicModel()
        self.request_count = 0
    
    def validate_coordinates(self, coordinates, n_wecs):
        """Validate coordinate input"""
        if len(coordinates) != n_wecs * 2:
            return False, f"Expected {n_wecs * 2} coordinates, got {len(coordinates)}"
        
        # Check for reasonable coordinate ranges
        positions = np.array(coordinates).reshape(n_wecs, 2)
        if np.any(positions < 0) or np.any(positions > 10000):
            return False, "Coordinates should be between 0 and 10000 meters"
        
        return True, "Valid"
    
    def predict_power(self, coordinates, n_wecs):
        """Simplified power prediction (replace with actual ML model)"""
        positions = np.array(coordinates).reshape(n_wecs, 2)
        
        # Calculate basic spatial features
        from scipy.spatial.distance import pdist
        distances = pdist(positions, 'euclidean')
        
        if len(distances) == 0:
            return 0
        
        mean_distance = np.mean(distances)
        min_distance = np.min(distances)
        
        # Simple heuristic model (replace with trained ML model)
        base_power = n_wecs * 80000  # 80kW per WEC
        
        # Distance factor (optimal around 100-150m)
        distance_factor = min(mean_distance / 120, 2.0)
        
        # Penalty for too close spacing
        if min_distance < 50:
            distance_factor *= 0.5
        
        return base_power * distance_factor

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict_power():
    """Predict power output for given WEC coordinates"""
    try:
        data = request.get_json()
        
        # Validate input
        if 'coordinates' not in data or 'n_wecs' not in data:
            return jsonify({'error': 'Missing coordinates or n_wecs parameter'}), 400
        
        coordinates = data['coordinates']
        n_wecs = data['n_wecs']
        
        # Validate coordinates
        api = WECOptimizationAPI()
        valid, message = api.validate_coordinates(coordinates, n_wecs)
        if not valid:
            return jsonify({'error': message}), 400
        
        # Predict power
        power_output = api.predict_power(coordinates, n_wecs)
        
        # Calculate basic metrics
        positions = np.array(coordinates).reshape(n_wecs, 2)
        from scipy.spatial.distance import pdist
        distances = pdist(positions, 'euclidean')
        
        metrics = {
            'power_output_watts': float(power_output),
            'power_output_mw': float(power_output / 1e6),
            'mean_distance': float(np.mean(distances)) if len(distances) > 0 else 0,
            'min_distance': float(np.min(distances)) if len(distances) > 0 else 0,
            'max_distance': float(np.max(distances)) if len(distances) > 0 else 0,
            'n_wecs': n_wecs
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/economics', methods=['POST'])
def economic_analysis():
    """Perform economic analysis for given layout"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['coordinates', 'n_wecs', 'power_output']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing {field} parameter'}), 400
        
        coordinates = data['coordinates']
        n_wecs = data['n_wecs']
        power_output = data['power_output']
        
        # Optional economic parameters
        economic_params = data.get('economic_params', {})
        
        # Update economic model if custom parameters provided
        if economic_params:
            for param, value in economic_params.items():
                if hasattr(economic_model, param):
                    setattr(economic_model, param, value)
        
        # Validate coordinates
        api = WECOptimizationAPI()
        valid, message = api.validate_coordinates(coordinates, n_wecs)
        if not valid:
            return jsonify({'error': message}), 400
        
        # Calculate economics
        _, economics = economic_model.get_economic_summary(
            np.array(coordinates), n_wecs, power_output
        )
        
        # Format response
        result = {
            'success': True,
            'economics': {
                'npv': float(economics['npv']),
                'roi_percent': float(economics['roi_percent']),
                'payback_period_years': float(economics['payback_period_years']),
                'annual_revenue': float(economics['annual_revenue']),
                'annual_costs': float(economics['annual_costs']),
                'total_installation_cost': float(economics['total_installation_cost']),
                'annual_energy_kwh': float(economics['energy_data']['annual_energy_kwh'])
            },
            'cost_breakdown': {
                'wec_costs': float(economics['installation_breakdown']['wec_costs']),
                'installation_costs': float(economics['installation_breakdown']['installation_costs']),
                'cable_costs': float(economics['installation_breakdown']['cable_costs']),
                'grid_connection': float(economics['installation_breakdown']['grid_connection'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Economic analysis error: {str(e)}")
        return jsonify({'error': f'Economic analysis failed: {str(e)}'}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_layout():
    """Optimize WEC layout for given objectives"""
    try:
        data = request.get_json()
        
        # Parameters
        n_wecs = data.get('n_wecs', 49)
        objective = data.get('objective', 'power')  # 'power', 'economics', 'combined'
        bounds = data.get('bounds', {'x_max': 1500, 'y_max': 1400, 'min_distance': 50})
        iterations = data.get('iterations', 10)
        
        # Generate optimized layout
        best_score = -float('inf')
        best_layout = None
        best_metrics = None
        
        api = WECOptimizationAPI()
        
        for i in range(iterations):
            # Generate random layout
            coordinates = generate_random_layout(n_wecs, bounds)
            
            # Calculate power
            power = api.predict_power(coordinates, n_wecs)
            
            if objective == 'power':
                score = power
            elif objective == 'economics':
                _, economics = economic_model.get_economic_summary(
                    np.array(coordinates), n_wecs, power
                )
                score = economics['npv']
            else:  # combined
                _, economics = economic_model.get_economic_summary(
                    np.array(coordinates), n_wecs, power
                )
                # Normalize and combine
                score = (power / 5e6) * 0.6 + (economics['npv'] / 100e6) * 0.4
            
            if score > best_score:
                best_score = score
                best_layout = coordinates
                best_metrics = {
                    'power_output': power,
                    'coordinates': coordinates.tolist()
                }
        
        if best_layout is not None:
            # Calculate final metrics for best layout
            positions = best_layout.reshape(n_wecs, 2)
            from scipy.spatial.distance import pdist
            distances = pdist(positions, 'euclidean')
            
            result = {
                'success': True,
                'optimized_layout': {
                    'coordinates': best_layout.tolist(),
                    'power_output_watts': float(best_metrics['power_output']),
                    'power_output_mw': float(best_metrics['power_output'] / 1e6),
                    'objective_score': float(best_score),
                    'mean_distance': float(np.mean(distances)),
                    'min_distance': float(np.min(distances)),
                    'n_wecs': n_wecs
                },
                'optimization_params': {
                    'objective': objective,
                    'iterations': iterations,
                    'bounds': bounds
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Optimization failed to find valid layout'}), 500
            
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500

def generate_random_layout(n_wecs, bounds):
    """Generate random WEC layout within bounds"""
    max_attempts = 100
    
    for _ in range(max_attempts):
        # Generate random positions
        x_coords = np.random.uniform(0, bounds['x_max'], n_wecs)
        y_coords = np.random.uniform(0, bounds['y_max'], n_wecs)
        
        positions = np.column_stack([x_coords, y_coords])
        
        # Check minimum distance constraint
        from scipy.spatial.distance import pdist
        distances = pdist(positions, 'euclidean')
        
        if len(distances) == 0 or np.all(distances >= bounds['min_distance']):
            coordinates = positions.flatten()
            return coordinates
    
    # Fallback: grid layout
    grid_size = int(np.ceil(np.sqrt(n_wecs)))
    spacing = max(bounds['min_distance'] * 1.5, 100)
    
    x_positions = []
    y_positions = []
    
    for i in range(n_wecs):
        row = i // grid_size
        col = i % grid_size
        x_positions.append(col * spacing)
        y_positions.append(row * spacing)
    
    positions = np.column_stack([x_positions, y_positions])
    return positions.flatten()

@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """API documentation"""
    docs = {
        'title': 'Wave Energy Farm Optimization API',
        'version': '1.0.0',
        'endpoints': {
            '/api/health': {
                'method': 'GET',
                'description': 'Health check endpoint',
                'response': 'Status and timestamp'
            },
            '/api/predict': {
                'method': 'POST',
                'description': 'Predict power output for WEC coordinates',
                'parameters': {
                    'coordinates': 'List of [x1,y1,x2,y2,...] coordinates',
                    'n_wecs': 'Number of WECs'
                },
                'response': 'Power prediction and layout metrics'
            },
            '/api/economics': {
                'method': 'POST',
                'description': 'Economic analysis for layout',
                'parameters': {
                    'coordinates': 'WEC coordinates',
                    'n_wecs': 'Number of WECs',
                    'power_output': 'Power output in watts',
                    'economic_params': 'Optional economic parameters'
                },
                'response': 'NPV, ROI, payback period, cost breakdown'
            },
            '/api/optimize': {
                'method': 'POST',
                'description': 'Optimize WEC layout',
                'parameters': {
                    'n_wecs': 'Number of WECs (default: 49)',
                    'objective': 'power/economics/combined (default: power)',
                    'bounds': 'Layout bounds and constraints',
                    'iterations': 'Optimization iterations (default: 10)'
                },
                'response': 'Optimized layout and performance metrics'
            }
        },
        'example_usage': {
            'predict': {
                'url': '/api/predict',
                'payload': {
                    'coordinates': [100, 100, 200, 200, 300, 300],
                    'n_wecs': 3
                }
            }
        }
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
