# Configuration file for Wave Energy Farm Optimization
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms"""
    
    # Genetic Algorithm Parameters
    ga_population_size: int = 20
    ga_generations: int = 30
    ga_crossover_prob: float = 0.7
    ga_mutation_prob: float = 0.2
    ga_tournament_size: int = 3
    
    # Bayesian Optimization Parameters
    bo_n_calls: int = 50
    bo_n_initial: int = 10
    bo_acquisition_function: str = 'EI'  # Expected Improvement
    
    # Layout Constraints
    x_min: float = 0.0
    x_max: float = 1500.0
    y_min: float = 0.0
    y_max: float = 1400.0
    min_distance: float = 50.0  # Minimum distance between WECs (meters)
    
    # Optimization Objectives
    power_weight: float = 0.6
    economics_weight: float = 0.4

@dataclass
class EconomicConfig:
    """Configuration for economic analysis"""
    
    # Costs (USD)
    wec_unit_cost: float = 2_000_000
    cable_cost_per_meter: float = 1_500
    installation_cost_per_wec: float = 500_000
    annual_maintenance_rate: float = 0.05
    grid_connection_cost: float = 10_000_000
    
    # Revenue
    electricity_price: float = 0.12  # $/kWh
    capacity_factor: float = 0.35
    power_transmission_efficiency: float = 0.95
    
    # Financial
    discount_rate: float = 0.08
    project_lifetime: int = 25
    hours_per_year: int = 8760

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    
    # Model Parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    
    nn_hidden_layers: tuple = (100, 50, 25)
    nn_activation: str = 'relu'
    nn_solver: str = 'adam'
    nn_alpha: float = 0.001
    nn_max_iter: int = 500
    
    # Training Parameters
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    
    # Plot Settings
    figure_width: int = 12
    figure_height: int = 8
    marker_size: int = 100
    line_width: int = 2
    alpha: float = 0.7
    
    # Colors
    primary_color: str = '#1f77b4'
    secondary_color: str = '#ff7f0e'
    success_color: str = '#2ca02c'
    warning_color: str = '#d62728'
    
    # 3D Plot Settings
    camera_eye: Dict[str, float] = None
    
    def __post_init__(self):
        if self.camera_eye is None:
            self.camera_eye = {'x': 1.5, 'y': 1.5, 'z': 1.2}

@dataclass
class APIConfig:
    """Configuration for API settings"""
    
    # Server Settings
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = False
    
    # Rate Limiting
    max_requests_per_minute: int = 100
    
    # Cache Settings
    cache_timeout: int = 3600  # 1 hour
    
    # Default Values
    default_n_wecs: int = 49
    default_objective: str = 'power'
    default_iterations: int = 10

class ConfigManager:
    """Manages configuration loading and saving"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.optimization = OptimizationConfig()
        self.economics = EconomicConfig()
        self.model = ModelConfig()
        self.visualization = VisualizationConfig()
        self.api = APIConfig()
        
        # Try to load existing configuration
        self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'optimization' in config_data:
                self.optimization = OptimizationConfig(**config_data['optimization'])
            if 'economics' in config_data:
                self.economics = EconomicConfig(**config_data['economics'])
            if 'model' in config_data:
                self.model = ModelConfig(**config_data['model'])
            if 'visualization' in config_data:
                self.visualization = VisualizationConfig(**config_data['visualization'])
            if 'api' in config_data:
                self.api = APIConfig(**config_data['api'])
                
            print(f"✅ Configuration loaded from {self.config_file}")
            
        except FileNotFoundError:
            print(f"⚠️ Configuration file {self.config_file} not found. Using defaults.")
            self.save_config()  # Create default config file
        except Exception as e:
            print(f"❌ Error loading configuration: {e}. Using defaults.")
    
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            config_data = {
                'optimization': asdict(self.optimization),
                'economics': asdict(self.economics),
                'model': asdict(self.model),
                'visualization': asdict(self.visualization),
                'api': asdict(self.api)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration saved to {self.config_file}")
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def update_optimization_config(self, **kwargs):
        """Update optimization configuration"""
        for key, value in kwargs.items():
            if hasattr(self.optimization, key):
                setattr(self.optimization, key, value)
        self.save_config()
    
    def update_economic_config(self, **kwargs):
        """Update economic configuration"""
        for key, value in kwargs.items():
            if hasattr(self.economics, key):
                setattr(self.economics, key, value)
        self.save_config()
    
    def get_optimization_bounds(self):
        """Get optimization bounds as dictionary"""
        return {
            'x_min': self.optimization.x_min,
            'x_max': self.optimization.x_max,
            'y_min': self.optimization.y_min,
            'y_max': self.optimization.y_max,
            'min_distance': self.optimization.min_distance
        }
    
    def get_economic_params(self):
        """Get economic parameters as dictionary"""
        return asdict(self.economics)
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("\n🔧 Current Configuration Summary:")
        print("=" * 50)
        
        print("\n📊 Optimization:")
        print(f"  GA Population: {self.optimization.ga_population_size}")
        print(f"  GA Generations: {self.optimization.ga_generations}")
        print(f"  BO Calls: {self.optimization.bo_n_calls}")
        print(f"  Min Distance: {self.optimization.min_distance}m")
        
        print("\n💰 Economics:")
        print(f"  WEC Cost: ${self.economics.wec_unit_cost:,}")
        print(f"  Electricity Price: ${self.economics.electricity_price}/kWh")
        print(f"  Project Lifetime: {self.economics.project_lifetime} years")
        print(f"  Discount Rate: {self.economics.discount_rate*100:.1f}%")
        
        print("\n🤖 Model:")
        print(f"  XGB Estimators: {self.model.xgb_n_estimators}")
        print(f"  RF Estimators: {self.model.rf_n_estimators}")
        print(f"  Test Size: {self.model.test_size*100:.0f}%")
        
        print("\n🌐 API:")
        print(f"  Host: {self.api.host}")
        print(f"  Port: {self.api.port}")
        print(f"  Max Requests/min: {self.api.max_requests_per_minute}")

# Create global configuration instance
config = ConfigManager()

# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    'api': {
        'debug': True,
        'port': 5000
    },
    'optimization': {
        'ga_generations': 10,  # Faster for development
        'bo_n_calls': 20
    }
}

PRODUCTION_CONFIG = {
    'api': {
        'debug': False,
        'port': 8080,
        'max_requests_per_minute': 1000
    },
    'optimization': {
        'ga_generations': 50,  # More thorough for production
        'bo_n_calls': 100
    }
}

def load_environment_config(environment: str = 'development'):
    """Load environment-specific configuration"""
    if environment == 'development':
        env_config = DEVELOPMENT_CONFIG
    elif environment == 'production':
        env_config = PRODUCTION_CONFIG
    else:
        print(f"⚠️ Unknown environment '{environment}'. Using default config.")
        return
    
    # Update global config
    for section, params in env_config.items():
        if section == 'optimization':
            config.update_optimization_config(**params)
        elif section == 'economics':
            config.update_economic_config(**params)
        # Add other sections as needed
    
    print(f"✅ Loaded {environment} configuration")

if __name__ == "__main__":
    # Example usage
    config.print_config_summary()
    
    # Update some parameters
    config.update_economic_config(
        electricity_price=0.15,
        project_lifetime=30
    )
    
    print("\n🔄 After updates:")
    config.print_config_summary()
