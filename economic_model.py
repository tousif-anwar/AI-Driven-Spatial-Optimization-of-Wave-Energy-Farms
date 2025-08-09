# Economic Model for Wave Energy Farm Optimization
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

class WECEconomicModel:
    """
    Economic model for Wave Energy Converter farms including:
    - Installation costs
    - Cable routing costs
    - Maintenance costs
    - Revenue from power generation
    - Net Present Value (NPV) calculation
    """
    
    def __init__(self):
        # Cost parameters (USD)
        self.wec_unit_cost = 2_000_000  # $2M per WEC unit
        self.cable_cost_per_meter = 1_500  # $1,500 per meter of cable
        self.installation_cost_per_wec = 500_000  # $500K installation per WEC
        self.annual_maintenance_rate = 0.05  # 5% of capital cost annually
        
        # Revenue parameters
        self.electricity_price = 0.12  # $0.12 per kWh
        self.capacity_factor = 0.35  # 35% average capacity utilization
        self.hours_per_year = 8760
        
        # Financial parameters
        self.discount_rate = 0.08  # 8% discount rate
        self.project_lifetime = 25  # 25 years
        
        # Technical parameters
        self.power_transmission_efficiency = 0.95
        self.grid_connection_cost = 10_000_000  # $10M for grid connection
        
    def calculate_installation_costs(self, coordinates, n_wecs):
        """Calculate total installation costs"""
        # Basic WEC costs
        wec_costs = n_wecs * self.wec_unit_cost
        installation_costs = n_wecs * self.installation_cost_per_wec
        
        # Cable routing costs (simplified - connects to nearest neighbor + grid)
        cable_costs = self.calculate_cable_costs(coordinates, n_wecs)
        
        total_installation = wec_costs + installation_costs + cable_costs + self.grid_connection_cost
        
        return {
            'wec_costs': wec_costs,
            'installation_costs': installation_costs,
            'cable_costs': cable_costs,
            'grid_connection': self.grid_connection_cost,
            'total_installation': total_installation
        }
    
    def calculate_cable_costs(self, coordinates, n_wecs):
        """Calculate cable routing costs using minimum spanning tree approach"""
        positions = coordinates.reshape(n_wecs, 2)
        
        # Calculate distances between all WECs
        distances = pdist(positions, 'euclidean')
        
        # Simplified cable cost: assume each WEC connects to nearest neighbor
        # Plus connection to shore (assume shore at origin)
        total_cable_length = 0
        
        # Minimum spanning tree approximation
        # For simplicity, use average distance * n_wecs as approximation
        if len(distances) > 0:
            avg_distance = np.mean(distances)
            total_cable_length = avg_distance * n_wecs * 0.8  # 80% efficiency factor
        
        # Add connection to shore (distance to centroid)
        centroid = np.mean(positions, axis=0)
        shore_distance = np.linalg.norm(centroid)
        total_cable_length += shore_distance
        
        return total_cable_length * self.cable_cost_per_meter
    
    def calculate_annual_revenue(self, total_power_watts):
        """Calculate annual revenue from power generation"""
        # Convert watts to kW
        total_power_kw = total_power_watts / 1000
        
        # Annual energy generation
        annual_energy_kwh = (total_power_kw * 
                           self.capacity_factor * 
                           self.hours_per_year * 
                           self.power_transmission_efficiency)
        
        # Annual revenue
        annual_revenue = annual_energy_kwh * self.electricity_price
        
        return {
            'annual_energy_kwh': annual_energy_kwh,
            'annual_revenue': annual_revenue
        }
    
    def calculate_annual_costs(self, installation_costs):
        """Calculate annual operating and maintenance costs"""
        return installation_costs['total_installation'] * self.annual_maintenance_rate
    
    def calculate_npv(self, coordinates, n_wecs, total_power_watts):
        """Calculate Net Present Value of the wave farm"""
        # Installation costs (year 0)
        installation = self.calculate_installation_costs(coordinates, n_wecs)
        
        # Annual revenue and costs
        revenue_data = self.calculate_annual_revenue(total_power_watts)
        annual_revenue = revenue_data['annual_revenue']
        annual_costs = self.calculate_annual_costs(installation)
        annual_cash_flow = annual_revenue - annual_costs
        
        # NPV calculation
        npv = -installation['total_installation']  # Initial investment
        
        for year in range(1, self.project_lifetime + 1):
            discounted_cash_flow = annual_cash_flow / ((1 + self.discount_rate) ** year)
            npv += discounted_cash_flow
        
        # Calculate additional metrics
        roi = (npv / installation['total_installation']) * 100
        payback_period = installation['total_installation'] / annual_cash_flow if annual_cash_flow > 0 else float('inf')
        
        return {
            'npv': npv,
            'roi_percent': roi,
            'payback_period_years': payback_period,
            'annual_cash_flow': annual_cash_flow,
            'total_installation_cost': installation['total_installation'],
            'annual_revenue': annual_revenue,
            'annual_costs': annual_costs,
            'installation_breakdown': installation,
            'energy_data': revenue_data
        }
    
    def optimize_for_economics(self, coordinates, n_wecs, total_power_watts):
        """
        Economic optimization objective function
        Returns negative NPV (for minimization algorithms)
        """
        economics = self.calculate_npv(coordinates, n_wecs, total_power_watts)
        return -economics['npv']  # Negative for minimization
    
    def get_economic_summary(self, coordinates, n_wecs, total_power_watts):
        """Get comprehensive economic summary"""
        economics = self.calculate_npv(coordinates, n_wecs, total_power_watts)
        
        summary = f"""
        📊 Economic Analysis Summary
        ═══════════════════════════════════════
        💰 Net Present Value: ${economics['npv']:,.0f}
        📈 ROI: {economics['roi_percent']:.1f}%
        ⏱️  Payback Period: {economics['payback_period_years']:.1f} years
        
        💵 Financial Flows (Annual):
        • Revenue: ${economics['annual_revenue']:,.0f}
        • O&M Costs: ${economics['annual_costs']:,.0f}
        • Net Cash Flow: ${economics['annual_cash_flow']:,.0f}
        
        🏗️ Installation Costs:
        • WEC Units: ${economics['installation_breakdown']['wec_costs']:,.0f}
        • Installation: ${economics['installation_breakdown']['installation_costs']:,.0f}
        • Cables: ${economics['installation_breakdown']['cable_costs']:,.0f}
        • Grid Connection: ${economics['installation_breakdown']['grid_connection']:,.0f}
        • Total: ${economics['total_installation_cost']:,.0f}
        
        ⚡ Energy Production:
        • Annual Generation: {economics['energy_data']['annual_energy_kwh']:,.0f} kWh
        • Capacity Factor: {self.capacity_factor*100:.1f}%
        • Power Output: {total_power_watts/1e6:.2f} MW
        """
        
        return summary, economics

# Enhanced WEC Optimizer with Economics
class EconomicWECOptimizer:
    """Enhanced optimizer that considers both power and economics"""
    
    def __init__(self, power_model, economic_model, feature_names, n_wecs, location='Perth'):
        self.power_model = power_model
        self.economic_model = economic_model
        self.feature_names = feature_names
        self.n_wecs = n_wecs
        self.location = location
        
        # Optimization weights
        self.power_weight = 0.6  # 60% focus on power
        self.economics_weight = 0.4  # 40% focus on economics
        
        # Bounds
        self.bounds = {
            'x_min': 0, 'x_max': 1500,
            'y_min': 0, 'y_max': 1400,
            'min_distance': 50,
        }
    
    def multi_objective_function(self, coordinates):
        """Combined power and economic optimization"""
        try:
            # Get power prediction
            power_prediction = self.predict_power(coordinates)
            if power_prediction < 0:
                return -1e10
            
            # Get economic metrics
            economics = self.economic_model.calculate_npv(coordinates, self.n_wecs, power_prediction)
            
            # Normalize and combine objectives
            # Normalize power (higher is better)
            normalized_power = power_prediction / 5e6  # Normalize to ~5MW typical max
            
            # Normalize NPV (higher is better, but can be negative)
            normalized_npv = economics['npv'] / 100e6  # Normalize to $100M scale
            
            # Combined objective (maximize both)
            combined_score = (self.power_weight * normalized_power + 
                            self.economics_weight * normalized_npv)
            
            return combined_score
            
        except Exception as e:
            return -1e10
    
    def predict_power(self, coordinates):
        """Predict power output (simplified - would use actual ML model)"""
        # This is a placeholder - in practice, use your trained ML model
        positions = coordinates.reshape(self.n_wecs, 2)
        distances = pdist(positions, 'euclidean')
        
        if np.any(distances < self.bounds['min_distance']):
            return -1  # Invalid configuration
        
        # Simplified power model based on distance
        mean_distance = np.mean(distances)
        area = self.calculate_area(positions)
        density = self.n_wecs / area if area > 0 else 0
        
        # Simple heuristic (replace with actual ML model)
        base_power = self.n_wecs * 80000  # 80kW per WEC baseline
        distance_factor = min(mean_distance / 100, 2.0)  # Optimal around 100m
        density_penalty = max(0, 1 - density * 1e6)  # Penalty for high density
        
        return base_power * distance_factor * density_penalty
    
    def calculate_area(self, positions):
        """Calculate convex hull area"""
        try:
            if len(positions) < 3:
                return 1e6  # Default area
            hull = ConvexHull(positions)
            return hull.volume
        except:
            return 1e6

# Example usage and testing
if __name__ == "__main__":
    # Test the economic model
    economic_model = WECEconomicModel()
    
    # Sample coordinates for 49 WECs
    np.random.seed(42)
    sample_coords = np.random.uniform(0, 1000, 49 * 2)
    sample_power = 4e6  # 4 MW
    
    # Calculate economics
    summary, economics = economic_model.get_economic_summary(sample_coords, 49, sample_power)
    print(summary)
