# Real-Time Wave Energy Farm Optimization
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import websockets
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import threading
import time
from queue import Queue

@dataclass
class WaveCondition:
    """Real-time wave condition data"""
    timestamp: datetime
    significant_wave_height: float  # meters
    peak_period: float  # seconds
    wave_direction: float  # degrees
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    location: str
    
@dataclass
class WECStatus:
    """Individual WEC status and performance"""
    wec_id: int
    position: tuple  # (x, y)
    power_output: float  # watts
    efficiency: float  # 0-1
    maintenance_required: bool
    last_updated: datetime
    operational: bool

class RealTimeOptimizer:
    """
    Real-time optimization system that adapts WEC configurations
    based on changing wave conditions and operational status
    """
    
    def __init__(self, initial_layout, predictor_model):
        self.current_layout = initial_layout
        self.predictor = predictor_model
        self.wave_conditions = Queue(maxsize=100)
        self.wec_statuses = {}
        self.optimization_history = []
        self.running = False
        
        # Adaptation parameters
        self.adaptation_threshold = 0.05  # 5% power improvement threshold
        self.update_interval = 300  # 5 minutes
        self.max_position_change = 50  # max 50m movement per update
        
    def start_monitoring(self):
        """Start real-time monitoring and optimization"""
        self.running = True
        
        # Start background threads
        threading.Thread(target=self._wave_data_collector, daemon=True).start()
        threading.Thread(target=self._optimization_loop, daemon=True).start()
        threading.Thread(target=self._performance_monitor, daemon=True).start()
        
        print("🌊 Real-time optimization system started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        print("⏹️ Real-time optimization system stopped")
    
    def _wave_data_collector(self):
        """Simulate real-time wave data collection"""
        while self.running:
            # Simulate wave condition updates (would be from real sensors/API)
            wave_data = WaveCondition(
                timestamp=datetime.now(),
                significant_wave_height=np.random.normal(2.5, 0.5),
                peak_period=np.random.normal(8.0, 1.0),
                wave_direction=np.random.normal(270, 30),
                wind_speed=np.random.normal(12, 3),
                wind_direction=np.random.normal(280, 20),
                location="Perth"
            )
            
            self.wave_conditions.put(wave_data)
            time.sleep(60)  # Update every minute
    
    def _optimization_loop(self):
        """Main optimization loop"""
        while self.running:
            try:
                # Get latest wave conditions
                if not self.wave_conditions.empty():
                    current_conditions = self.wave_conditions.get()
                    
                    # Evaluate current layout performance
                    current_power = self._evaluate_layout(self.current_layout, current_conditions)
                    
                    # Generate adaptive layout suggestions
                    optimized_layout = self._adaptive_optimization(current_conditions)
                    optimized_power = self._evaluate_layout(optimized_layout, current_conditions)
                    
                    # Check if adaptation is beneficial
                    improvement = (optimized_power - current_power) / current_power
                    
                    if improvement > self.adaptation_threshold:
                        # Implement gradual layout changes
                        new_layout = self._gradual_adaptation(self.current_layout, optimized_layout)
                        
                        self.optimization_history.append({
                            'timestamp': datetime.now(),
                            'old_layout': self.current_layout.copy(),
                            'new_layout': new_layout.copy(),
                            'wave_conditions': current_conditions,
                            'power_improvement': improvement,
                            'action': 'layout_update'
                        })
                        
                        self.current_layout = new_layout
                        print(f"🔄 Layout adapted - Power improvement: {improvement:.2%}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Optimization error: {e}")
                time.sleep(30)
    
    def _performance_monitor(self):
        """Monitor individual WEC performance"""
        while self.running:
            try:
                # Update WEC statuses (simulate sensor data)
                for i, (x, y) in enumerate(self.current_layout):
                    # Simulate performance metrics
                    base_power = 45000
                    efficiency = np.random.normal(0.85, 0.1)
                    maintenance_prob = np.random.random()
                    
                    self.wec_statuses[i] = WECStatus(
                        wec_id=i,
                        position=(x, y),
                        power_output=base_power * efficiency,
                        efficiency=efficiency,
                        maintenance_required=maintenance_prob < 0.02,  # 2% chance
                        last_updated=datetime.now(),
                        operational=efficiency > 0.3
                    )
                
                # Check for maintenance requirements
                maintenance_needed = [wec for wec in self.wec_statuses.values() 
                                    if wec.maintenance_required]
                
                if maintenance_needed:
                    print(f"⚠️ Maintenance required for {len(maintenance_needed)} WECs")
                    self._handle_maintenance(maintenance_needed)
                
                time.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)
    
    def _evaluate_layout(self, layout, wave_conditions):
        """Evaluate layout performance under specific wave conditions"""
        # Convert layout to coordinates
        coordinates = np.array(layout).flatten()
        n_wecs = len(layout)
        
        # Base prediction
        base_power = self.predictor.predict(coordinates, n_wecs)
        
        # Apply wave condition modifiers
        wave_modifier = self._calculate_wave_modifier(wave_conditions)
        
        return base_power * wave_modifier
    
    def _calculate_wave_modifier(self, wave_conditions):
        """Calculate power modifier based on wave conditions"""
        # Optimal wave height: 2-3 meters
        height_factor = 1.0
        if wave_conditions.significant_wave_height < 1.5:
            height_factor = 0.6  # Too low
        elif wave_conditions.significant_wave_height > 4.0:
            height_factor = 0.7  # Too high, safety limits
        elif 2.0 <= wave_conditions.significant_wave_height <= 3.0:
            height_factor = 1.2  # Optimal range
        
        # Period factor (optimal around 8-10 seconds)
        period_factor = 1.0
        if 7.0 <= wave_conditions.peak_period <= 10.0:
            period_factor = 1.1
        elif wave_conditions.peak_period < 5.0 or wave_conditions.peak_period > 15.0:
            period_factor = 0.8
        
        # Wind factor (helpful up to certain point)
        wind_factor = 1.0 + min(wave_conditions.wind_speed * 0.01, 0.15)
        
        return height_factor * period_factor * wind_factor
    
    def _adaptive_optimization(self, wave_conditions):
        """Generate layout optimized for current conditions"""
        # For this implementation, we'll use a simplified approach
        # In practice, this would use sophisticated optimization algorithms
        
        current_layout = np.array(self.current_layout)
        n_wecs = len(current_layout)
        
        # Adjust spacing based on wave direction and characteristics
        optimal_spacing = self._calculate_optimal_spacing(wave_conditions)
        
        # Generate candidate layouts with adjusted spacing
        best_layout = current_layout.copy()
        best_power = self._evaluate_layout(best_layout.tolist(), wave_conditions)
        
        for _ in range(10):  # Try 10 variations
            candidate = self._adjust_layout_spacing(current_layout, optimal_spacing)
            candidate_power = self._evaluate_layout(candidate.tolist(), wave_conditions)
            
            if candidate_power > best_power:
                best_power = candidate_power
                best_layout = candidate
        
        return best_layout.tolist()
    
    def _calculate_optimal_spacing(self, wave_conditions):
        """Calculate optimal WEC spacing for current wave conditions"""
        # Base spacing on wavelength
        wave_length = 1.56 * wave_conditions.peak_period ** 2  # Deep water approximation
        
        # Optimal spacing is typically 2-3 wavelengths
        optimal_spacing = wave_length * 2.5
        
        # Adjust for wave height
        if wave_conditions.significant_wave_height > 3.0:
            optimal_spacing *= 1.2  # Increase spacing for large waves
        elif wave_conditions.significant_wave_height < 2.0:
            optimal_spacing *= 0.8  # Decrease spacing for small waves
        
        return max(50, min(200, optimal_spacing))  # Constrain to reasonable bounds
    
    def _adjust_layout_spacing(self, layout, target_spacing):
        """Adjust layout to achieve target spacing"""
        adjusted = layout.copy()
        n_wecs = len(layout)
        
        # Calculate current center
        center_x = np.mean(layout[:, 0])
        center_y = np.mean(layout[:, 1])
        
        for i in range(n_wecs):
            # Vector from center to WEC
            dx = layout[i, 0] - center_x
            dy = layout[i, 1] - center_y
            
            current_distance = np.sqrt(dx**2 + dy**2)
            if current_distance > 0:
                # Scale to target spacing
                scale_factor = (target_spacing / 100) * 0.1  # Gradual adjustment
                adjusted[i, 0] = center_x + dx * (1 + scale_factor)
                adjusted[i, 1] = center_y + dy * (1 + scale_factor)
        
        # Ensure bounds
        adjusted[:, 0] = np.clip(adjusted[:, 0], 0, 1500)
        adjusted[:, 1] = np.clip(adjusted[:, 1], 0, 1400)
        
        return adjusted
    
    def _gradual_adaptation(self, current_layout, target_layout):
        """Implement gradual changes to avoid operational disruption"""
        current = np.array(current_layout)
        target = np.array(target_layout)
        
        # Calculate movement vectors
        movement = target - current
        
        # Limit movement per update
        movement_magnitude = np.linalg.norm(movement, axis=1)
        for i in range(len(movement)):
            if movement_magnitude[i] > self.max_position_change:
                scale = self.max_position_change / movement_magnitude[i]
                movement[i] *= scale
        
        new_layout = current + movement
        
        # Ensure minimum distance constraints
        new_layout = self._enforce_distance_constraints(new_layout)
        
        return new_layout.tolist()
    
    def _enforce_distance_constraints(self, layout, min_distance=50):
        """Ensure minimum distance constraints are met"""
        adjusted = layout.copy()
        n_wecs = len(layout)
        
        for i in range(n_wecs):
            for j in range(i+1, n_wecs):
                distance = np.linalg.norm(adjusted[i] - adjusted[j])
                if distance < min_distance:
                    # Push WECs apart
                    direction = (adjusted[j] - adjusted[i]) / distance
                    push_distance = (min_distance - distance) / 2
                    adjusted[i] -= direction * push_distance
                    adjusted[j] += direction * push_distance
        
        return adjusted
    
    def _handle_maintenance(self, maintenance_wecs):
        """Handle WECs requiring maintenance"""
        for wec in maintenance_wecs:
            # Temporarily mark as non-operational
            wec.operational = False
            
            # Log maintenance event
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'wec_id': wec.wec_id,
                'action': 'maintenance_scheduled',
                'position': wec.position,
                'efficiency': wec.efficiency
            })
            
            print(f"🔧 WEC {wec.wec_id} scheduled for maintenance")
    
    def get_real_time_status(self):
        """Get current system status"""
        operational_wecs = sum(1 for wec in self.wec_statuses.values() if wec.operational)
        total_power = sum(wec.power_output for wec in self.wec_statuses.values() if wec.operational)
        avg_efficiency = np.mean([wec.efficiency for wec in self.wec_statuses.values() if wec.operational])
        
        return {
            'timestamp': datetime.now(),
            'operational_wecs': operational_wecs,
            'total_wecs': len(self.wec_statuses),
            'total_power_mw': total_power / 1e6,
            'average_efficiency': avg_efficiency,
            'maintenance_required': sum(1 for wec in self.wec_statuses.values() if wec.maintenance_required),
            'layout_updates': len([h for h in self.optimization_history if h.get('action') == 'layout_update']),
            'current_layout': self.current_layout
        }
    
    def export_performance_data(self, filename=None):
        """Export performance history to file"""
        if filename is None:
            filename = f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'optimization_history': [
                {**entry, 'timestamp': entry['timestamp'].isoformat()}
                for entry in self.optimization_history
            ],
            'final_status': self.get_real_time_status()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📊 Performance data exported to {filename}")

class PredictiveMaintenanceSystem:
    """Predictive maintenance using machine learning"""
    
    def __init__(self):
        self.failure_model = None
        self.maintenance_schedule = {}
        
    def train_failure_prediction(self, historical_data):
        """Train model to predict WEC failures"""
        # This would use historical performance data, weather conditions,
        # and maintenance records to predict when WECs need maintenance
        pass
    
    def predict_maintenance_needs(self, wec_statuses, wave_conditions):
        """Predict maintenance requirements"""
        maintenance_predictions = {}
        
        for wec_id, status in wec_statuses.items():
            # Simple rule-based prediction (would be ML in practice)
            risk_score = 0
            
            if status.efficiency < 0.7:
                risk_score += 0.3
            
            if status.power_output < 30000:  # Below expected
                risk_score += 0.2
            
            # Add environmental stress factors
            risk_score += min(wave_conditions.significant_wave_height / 5.0, 0.3)
            risk_score += min(wave_conditions.wind_speed / 30.0, 0.2)
            
            maintenance_predictions[wec_id] = {
                'risk_score': risk_score,
                'recommended_action': 'inspect' if risk_score > 0.4 else 'monitor',
                'urgency': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low'
            }
        
        return maintenance_predictions

# Example usage
if __name__ == "__main__":
    # Initialize with sample layout
    sample_layout = [(100 + i*120, 100 + j*120) for i in range(7) for j in range(7)][:49]
    
    # Mock predictor (would be your actual trained model)
    class MockPredictor:
        def predict(self, coords, n_wecs):
            return n_wecs * 45000 * np.random.uniform(0.8, 1.2)
    
    predictor = MockPredictor()
    
    # Create and start real-time optimizer
    optimizer = RealTimeOptimizer(sample_layout, predictor)
    optimizer.start_monitoring()
    
    # Run for demo period
    print("🌊 Running real-time optimization demo...")
    time.sleep(10)  # Run for 10 seconds in demo
    
    # Get status and stop
    status = optimizer.get_real_time_status()
    print(f"📊 Final Status: {status}")
    
    optimizer.stop_monitoring()
    optimizer.export_performance_data("demo_performance.json")
