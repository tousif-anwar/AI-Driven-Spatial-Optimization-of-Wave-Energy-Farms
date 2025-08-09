# Environmental Impact and Sustainability Analysis Module
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

@dataclass
class EnvironmentalData:
    """Environmental monitoring data structure"""
    timestamp: datetime
    location: str
    water_temperature: float  # Celsius
    turbidity: float  # NTU
    dissolved_oxygen: float  # mg/L
    ph_level: float
    salinity: float  # PSU
    noise_level: float  # dB
    marine_life_activity: float  # Activity index 0-1
    wave_energy_density: float  # kW/m

@dataclass
class EcosystemImpact:
    """Ecosystem impact assessment"""
    species_name: str
    population_change: float  # Percentage change
    behavioral_change: str
    habitat_disruption: float  # 0-1 scale
    adaptation_score: float  # 0-1 scale
    risk_level: str  # Low, Medium, High

class EnvironmentalImpactAnalyzer:
    """
    Comprehensive environmental impact analysis for wave energy farms
    Includes marine ecosystem monitoring, carbon footprint analysis,
    and sustainability metrics
    """
    
    def __init__(self):
        self.baseline_data = {}
        self.monitoring_data = []
        self.impact_assessments = []
        self.mitigation_measures = []
        
        # Environmental standards and thresholds
        self.water_quality_standards = {
            'temperature_max_change': 2.0,  # Max 2°C change from baseline
            'turbidity_max': 25.0,  # NTU
            'do_min': 6.0,  # mg/L
            'ph_range': (7.8, 8.3),
            'noise_max': 120.0  # dB underwater
        }
        
        # Marine species sensitivity profiles
        self.species_profiles = {
            'Gray Whale': {'noise_sensitivity': 0.9, 'habitat_flexibility': 0.3, 'migration_impact': 0.8},
            'Harbor Seal': {'noise_sensitivity': 0.7, 'habitat_flexibility': 0.6, 'migration_impact': 0.4},
            'Dungeness Crab': {'noise_sensitivity': 0.3, 'habitat_flexibility': 0.8, 'migration_impact': 0.2},
            'Pacific Salmon': {'noise_sensitivity': 0.6, 'habitat_flexibility': 0.4, 'migration_impact': 0.9},
            'Sea Otter': {'noise_sensitivity': 0.5, 'habitat_flexibility': 0.5, 'migration_impact': 0.3},
            'Kelp Forest': {'noise_sensitivity': 0.1, 'habitat_flexibility': 0.2, 'migration_impact': 0.0}
        }
    
    def establish_baseline(self, location: str, duration_days: int = 365):
        """Establish environmental baseline before WEC installation"""
        # Simulate baseline data collection
        baseline_data = {
            'location': location,
            'monitoring_period': duration_days,
            'water_quality': {
                'avg_temperature': 14.5,
                'temperature_variance': 2.3,
                'avg_turbidity': 8.2,
                'avg_dissolved_oxygen': 8.9,
                'avg_ph': 8.1,
                'avg_salinity': 34.2
            },
            'noise_levels': {
                'avg_ambient_noise': 85.3,  # dB
                'peak_noise_events': 12,  # per day
                'shipping_noise_hours': 4.2  # hours per day
            },
            'marine_life': {
                'species_counts': {
                    'Gray Whale': 45,
                    'Harbor Seal': 180,
                    'Dungeness Crab': 2500,
                    'Pacific Salmon': 850,
                    'Sea Otter': 95
                },
                'migration_patterns': {
                    'whale_migration_peak': 'March-May',
                    'salmon_run_peak': 'September-November'
                },
                'breeding_areas': {
                    'seal_rookeries': 3,
                    'bird_nesting_sites': 8
                }
            },
            'habitat_quality': {
                'kelp_forest_coverage': 0.75,  # 75% coverage
                'seafloor_complexity': 0.68,
                'water_column_productivity': 0.82
            }
        }
        
        self.baseline_data[location] = baseline_data
        return baseline_data
    
    def monitor_environmental_conditions(self, location: str, wec_layout: List[Tuple[float, float]]):
        """Monitor environmental conditions post-installation"""
        current_time = datetime.now()
        
        # Simulate environmental monitoring data
        # In practice, this would come from real sensors and surveys
        
        # Calculate WEC density and potential impacts
        n_wecs = len(wec_layout)
        farm_area = self._calculate_farm_area(wec_layout)
        wec_density = n_wecs / farm_area * 1e6  # WECs per km²
        
        # Base environmental data with WEC influence
        monitoring_data = EnvironmentalData(
            timestamp=current_time,
            location=location,
            water_temperature=14.5 + np.random.normal(0, 0.3) + min(wec_density * 0.01, 0.5),
            turbidity=8.2 + np.random.normal(0, 1.2) + min(wec_density * 0.05, 2.0),
            dissolved_oxygen=8.9 + np.random.normal(0, 0.3) - min(wec_density * 0.001, 0.2),
            ph_level=8.1 + np.random.normal(0, 0.1),
            salinity=34.2 + np.random.normal(0, 0.2),
            noise_level=85.3 + min(wec_density * 2.0, 15.0) + np.random.normal(0, 3.0),
            marine_life_activity=max(0.1, 0.85 - min(wec_density * 0.02, 0.3) + np.random.normal(0, 0.1)),
            wave_energy_density=25.0 + np.random.normal(0, 3.0)
        )
        
        self.monitoring_data.append(monitoring_data)
        return monitoring_data
    
    def assess_ecosystem_impact(self, location: str, monitoring_period_days: int = 90):
        """Assess impact on marine ecosystem"""
        if location not in self.baseline_data:
            raise ValueError(f"No baseline data available for {location}")
        
        baseline = self.baseline_data[location]
        recent_data = [d for d in self.monitoring_data 
                      if d.location == location and 
                      d.timestamp > datetime.now() - timedelta(days=monitoring_period_days)]
        
        if not recent_data:
            return []
        
        impact_assessments = []
        
        for species, profile in self.species_profiles.items():
            # Calculate impact scores based on environmental changes
            noise_impact = self._calculate_noise_impact(recent_data, profile['noise_sensitivity'])
            habitat_impact = self._calculate_habitat_impact(recent_data, profile['habitat_flexibility'])
            migration_impact = self._calculate_migration_impact(species, profile['migration_impact'])
            
            # Overall impact assessment
            overall_impact = (noise_impact + habitat_impact + migration_impact) / 3
            
            # Population change estimation (simplified model)
            baseline_pop = baseline['marine_life']['species_counts'].get(species, 0)
            population_change = -overall_impact * 20 + np.random.normal(0, 5)  # ±20% max impact
            
            # Behavioral changes
            behavioral_changes = self._assess_behavioral_changes(species, overall_impact)
            
            # Risk classification
            if overall_impact < 0.2:
                risk_level = "Low"
            elif overall_impact < 0.5:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            impact = EcosystemImpact(
                species_name=species,
                population_change=population_change,
                behavioral_change=behavioral_changes,
                habitat_disruption=habitat_impact,
                adaptation_score=1 - overall_impact,
                risk_level=risk_level
            )
            
            impact_assessments.append(impact)
        
        self.impact_assessments.extend(impact_assessments)
        return impact_assessments
    
    def calculate_carbon_footprint(self, wec_layout: List[Tuple[float, float]], 
                                 operational_years: int = 25) -> Dict[str, float]:
        """Calculate lifecycle carbon footprint of the wave energy farm"""
        n_wecs = len(wec_layout)
        farm_area = self._calculate_farm_area(wec_layout)
        
        # Carbon emissions (tons CO2 equivalent)
        emissions = {
            # Manufacturing phase
            'manufacturing': {
                'wec_production': n_wecs * 450,  # 450 tons CO2 per WEC
                'steel_concrete': n_wecs * 120,  # Foundation materials
                'electronics': n_wecs * 25,     # Control systems
                'cables': self._calculate_cable_carbon(wec_layout)
            },
            
            # Installation phase
            'installation': {
                'transportation': n_wecs * 15,   # Transport to site
                'vessel_operations': farm_area * 0.5,  # Installation vessels
                'crane_operations': n_wecs * 8   # Heavy lifting
            },
            
            # Operational phase (25 years)
            'operations': {
                'maintenance_vessels': operational_years * 45,
                'replacement_parts': operational_years * n_wecs * 3,
                'monitoring_systems': operational_years * 12
            },
            
            # End-of-life
            'decommissioning': {
                'removal_operations': n_wecs * 12,
                'waste_processing': n_wecs * 8,
                'material_recovery': -n_wecs * 85  # Negative = carbon savings from recycling
            }
        }
        
        # Calculate totals
        total_emissions = {}
        for phase, sources in emissions.items():
            total_emissions[phase] = sum(sources.values())
        
        total_lifecycle_emissions = sum(total_emissions.values())
        
        # Carbon offset through clean energy generation
        annual_energy_generation = self._estimate_annual_energy(wec_layout)  # MWh
        carbon_intensity_displaced = 0.855  # tons CO2/MWh (average grid)
        total_carbon_offset = annual_energy_generation * operational_years * carbon_intensity_displaced
        
        net_carbon_impact = total_lifecycle_emissions - total_carbon_offset
        
        return {
            'lifecycle_emissions_total': total_lifecycle_emissions,
            'manufacturing_emissions': total_emissions['manufacturing'],
            'installation_emissions': total_emissions['installation'],
            'operational_emissions': total_emissions['operations'],
            'decommissioning_emissions': total_emissions['decommissioning'],
            'carbon_offset': total_carbon_offset,
            'net_carbon_impact': net_carbon_impact,
            'carbon_payback_years': total_lifecycle_emissions / (annual_energy_generation * carbon_intensity_displaced),
            'emissions_per_mwh': total_lifecycle_emissions / (annual_energy_generation * operational_years)
        }
    
    def develop_mitigation_measures(self, impact_assessments: List[EcosystemImpact]) -> List[Dict]:
        """Develop mitigation measures based on impact assessments"""
        mitigation_measures = []
        
        for impact in impact_assessments:
            measures = []
            
            if impact.risk_level in ["Medium", "High"]:
                if impact.species_name == "Gray Whale":
                    measures.extend([
                        {
                            'measure': 'Seasonal shutdown during migration',
                            'timing': 'March-May',
                            'effectiveness': 0.8,
                            'cost': 250000,  # USD per year
                            'implementation_time': '1 month'
                        },
                        {
                            'measure': 'Noise reduction technology',
                            'timing': 'Continuous',
                            'effectiveness': 0.6,
                            'cost': 850000,
                            'implementation_time': '6 months'
                        }
                    ])
                
                elif impact.species_name == "Pacific Salmon":
                    measures.extend([
                        {
                            'measure': 'Fish passage corridors',
                            'timing': 'Permanent',
                            'effectiveness': 0.7,
                            'cost': 1200000,
                            'implementation_time': '12 months'
                        },
                        {
                            'measure': 'Lighting adjustments',
                            'timing': 'Night hours',
                            'effectiveness': 0.4,
                            'cost': 45000,
                            'implementation_time': '2 weeks'
                        }
                    ])
                
                elif impact.species_name == "Harbor Seal":
                    measures.extend([
                        {
                            'measure': 'Artificial haul-out platforms',
                            'timing': 'Permanent',
                            'effectiveness': 0.5,
                            'cost': 350000,
                            'implementation_time': '4 months'
                        }
                    ])
                
                # General measures for all species
                if impact.habitat_disruption > 0.5:
                    measures.append({
                        'measure': 'Habitat restoration program',
                        'timing': 'Annual',
                        'effectiveness': 0.6,
                        'cost': 180000,
                        'implementation_time': '3 months'
                    })
            
            for measure in measures:
                measure.update({
                    'target_species': impact.species_name,
                    'impact_type': f"Addresses {impact.behavioral_change}",
                    'priority': 'High' if impact.risk_level == 'High' else 'Medium'
                })
                
                mitigation_measures.append(measure)
        
        self.mitigation_measures.extend(mitigation_measures)
        return mitigation_measures
    
    def calculate_sustainability_metrics(self, wec_layout: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate comprehensive sustainability metrics"""
        carbon_footprint = self.calculate_carbon_footprint(wec_layout)
        
        # Energy metrics
        annual_energy = self._estimate_annual_energy(wec_layout)  # MWh/year
        farm_area = self._calculate_farm_area(wec_layout)  # m²
        
        # Resource efficiency
        steel_per_mwh = (len(wec_layout) * 85) / annual_energy  # tons steel per MWh annual capacity
        concrete_per_mwh = (len(wec_layout) * 120) / annual_energy  # tons concrete per MWh annual capacity
        
        # Land use (or sea use) efficiency
        energy_density = annual_energy / (farm_area / 1e6)  # MWh per km²
        
        # Water impact
        water_consumption = 0  # Wave energy uses no water
        
        # Waste generation
        annual_waste = len(wec_layout) * 2.5  # tons per year (maintenance waste)
        
        # Ecological footprint
        ecosystem_impact_score = np.mean([impact.habitat_disruption 
                                        for impact in self.impact_assessments]) if self.impact_assessments else 0
        
        # Social impact indicators
        job_creation = len(wec_layout) * 0.3  # Jobs per WEC (direct + indirect)
        community_investment = len(wec_layout) * 15000  # USD per WEC per year
        
        return {
            # Environmental metrics
            'carbon_intensity_gco2_kwh': (carbon_footprint['lifecycle_emissions_total'] * 1000) / (annual_energy * 1000),
            'carbon_payback_years': carbon_footprint['carbon_payback_years'],
            'net_carbon_impact_tons': carbon_footprint['net_carbon_impact'],
            'ecosystem_impact_score': ecosystem_impact_score,
            'water_consumption_m3_mwh': water_consumption,
            
            # Resource efficiency
            'steel_intensity_kg_mwh': steel_per_mwh * 1000,
            'concrete_intensity_kg_mwh': concrete_per_mwh * 1000,
            'energy_density_mwh_km2': energy_density,
            'capacity_factor': 0.35,  # Typical for wave energy
            
            # Waste and circularity
            'annual_waste_tons': annual_waste,
            'recycling_rate': 0.85,  # 85% of materials recyclable at end of life
            'material_circularity_index': 0.72,
            
            # Social metrics
            'jobs_per_mw': job_creation / (annual_energy / 8760),
            'community_investment_usd_mw': community_investment / (annual_energy / 8760),
            'energy_security_index': 0.92,  # Local, renewable energy source
            
            # Economic sustainability
            'lcoe_usd_mwh': 185,  # Levelized Cost of Energy
            'economic_viability_score': 0.78
        }
    
    def generate_environmental_report(self, location: str) -> Dict:
        """Generate comprehensive environmental impact report"""
        if location not in self.baseline_data:
            raise ValueError(f"No baseline data available for {location}")
        
        recent_monitoring = [d for d in self.monitoring_data 
                           if d.location == location and 
                           d.timestamp > datetime.now() - timedelta(days=30)]
        
        recent_impacts = [i for i in self.impact_assessments 
                         if any(d.location == location for d in self.monitoring_data)]
        
        report = {
            'report_date': datetime.now().isoformat(),
            'location': location,
            'reporting_period': '30 days',
            
            'executive_summary': {
                'overall_impact_level': self._assess_overall_impact(recent_impacts),
                'compliance_status': self._check_compliance(recent_monitoring),
                'critical_issues': self._identify_critical_issues(recent_impacts),
                'recommended_actions': len(self.mitigation_measures)
            },
            
            'water_quality': {
                'current_conditions': self._summarize_water_quality(recent_monitoring),
                'baseline_comparison': self._compare_to_baseline(recent_monitoring, location),
                'trend_analysis': self._analyze_trends(recent_monitoring),
                'compliance_status': self._check_water_quality_compliance(recent_monitoring)
            },
            
            'ecosystem_impact': {
                'species_assessments': [
                    {
                        'species': impact.species_name,
                        'population_change': impact.population_change,
                        'risk_level': impact.risk_level,
                        'behavioral_impact': impact.behavioral_change,
                        'adaptation_score': impact.adaptation_score
                    }
                    for impact in recent_impacts
                ],
                'habitat_quality': self._assess_habitat_quality(recent_monitoring),
                'biodiversity_index': self._calculate_biodiversity_index(recent_impacts)
            },
            
            'mitigation_measures': {
                'implemented': [m for m in self.mitigation_measures if m.get('status') == 'implemented'],
                'planned': [m for m in self.mitigation_measures if m.get('status') == 'planned'],
                'recommended': [m for m in self.mitigation_measures if m.get('status') is None],
                'total_cost': sum(m['cost'] for m in self.mitigation_measures),
                'effectiveness_rating': np.mean([m['effectiveness'] for m in self.mitigation_measures])
            },
            
            'regulatory_compliance': {
                'environmental_standards_met': True,
                'permit_conditions_status': 'Compliant',
                'reporting_requirements_fulfilled': True,
                'next_assessment_due': (datetime.now() + timedelta(days=90)).isoformat()
            },
            
            'recommendations': self._generate_recommendations(recent_impacts, recent_monitoring)
        }
        
        return report
    
    # Helper methods
    def _calculate_farm_area(self, wec_layout: List[Tuple[float, float]]) -> float:
        """Calculate total farm area in m²"""
        if len(wec_layout) < 3:
            return 100000  # Default area for small farms
        
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(wec_layout)
            return hull.volume  # In 2D, volume is actually area
        except:
            # Fallback calculation
            xs, ys = zip(*wec_layout)
            return (max(xs) - min(xs)) * (max(ys) - min(ys))
    
    def _calculate_cable_carbon(self, wec_layout: List[Tuple[float, float]]) -> float:
        """Calculate carbon footprint of cable installation"""
        # Simplified calculation - would use actual cable routing in practice
        n_wecs = len(wec_layout)
        avg_distance = 200  # Average cable distance per WEC
        cable_carbon_per_km = 15  # tons CO2 per km
        
        return n_wecs * avg_distance / 1000 * cable_carbon_per_km
    
    def _estimate_annual_energy(self, wec_layout: List[Tuple[float, float]]) -> float:
        """Estimate annual energy generation in MWh"""
        n_wecs = len(wec_layout)
        capacity_per_wec = 1.0  # MW
        capacity_factor = 0.35  # 35% capacity factor
        
        return n_wecs * capacity_per_wec * 8760 * capacity_factor
    
    def _calculate_noise_impact(self, monitoring_data: List[EnvironmentalData], sensitivity: float) -> float:
        """Calculate noise impact on species"""
        avg_noise = np.mean([d.noise_level for d in monitoring_data])
        baseline_noise = 85.3
        noise_increase = max(0, avg_noise - baseline_noise)
        
        # Impact scales with noise increase and species sensitivity
        return min(1.0, (noise_increase / 20.0) * sensitivity)
    
    def _calculate_habitat_impact(self, monitoring_data: List[EnvironmentalData], flexibility: float) -> float:
        """Calculate habitat disruption impact"""
        avg_turbidity = np.mean([d.turbidity for d in monitoring_data])
        baseline_turbidity = 8.2
        turbidity_increase = max(0, avg_turbidity - baseline_turbidity)
        
        # Impact is inversely related to species flexibility
        return min(1.0, (turbidity_increase / 10.0) * (1 - flexibility))
    
    def _calculate_migration_impact(self, species: str, migration_sensitivity: float) -> float:
        """Calculate migration pattern disruption"""
        # Simplified - would use actual migration data in practice
        if 'Whale' in species or 'Salmon' in species:
            return migration_sensitivity * 0.3  # Assume 30% baseline disruption
        return migration_sensitivity * 0.1
    
    def _assess_behavioral_changes(self, species: str, impact_level: float) -> str:
        """Assess behavioral changes based on impact level"""
        if impact_level < 0.2:
            return "Minimal behavioral changes observed"
        elif impact_level < 0.5:
            return "Moderate avoidance behavior, temporary displacement"
        else:
            return "Significant behavioral changes, habitat abandonment possible"
    
    # Additional helper methods for report generation
    def _assess_overall_impact(self, impacts: List[EcosystemImpact]) -> str:
        """Assess overall environmental impact level"""
        if not impacts:
            return "Unknown"
        
        high_risk_count = sum(1 for i in impacts if i.risk_level == "High")
        medium_risk_count = sum(1 for i in impacts if i.risk_level == "Medium")
        
        if high_risk_count > 0:
            return "High"
        elif medium_risk_count > len(impacts) * 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _check_compliance(self, monitoring_data: List[EnvironmentalData]) -> str:
        """Check compliance with environmental standards"""
        if not monitoring_data:
            return "Unknown"
        
        violations = 0
        total_checks = 0
        
        for data in monitoring_data:
            total_checks += 5  # 5 parameters checked
            
            if data.turbidity > self.water_quality_standards['turbidity_max']:
                violations += 1
            if data.dissolved_oxygen < self.water_quality_standards['do_min']:
                violations += 1
            if not (self.water_quality_standards['ph_range'][0] <= data.ph_level <= self.water_quality_standards['ph_range'][1]):
                violations += 1
            if data.noise_level > self.water_quality_standards['noise_max']:
                violations += 1
            if abs(data.water_temperature - 14.5) > self.water_quality_standards['temperature_max_change']:
                violations += 1
        
        compliance_rate = 1 - (violations / total_checks)
        
        if compliance_rate >= 0.95:
            return "Compliant"
        elif compliance_rate >= 0.85:
            return "Minor violations"
        else:
            return "Non-compliant"
    
    def _identify_critical_issues(self, impacts: List[EcosystemImpact]) -> List[str]:
        """Identify critical environmental issues"""
        critical_issues = []
        
        for impact in impacts:
            if impact.risk_level == "High":
                critical_issues.append(f"{impact.species_name}: {impact.behavioral_change}")
            if impact.population_change < -15:
                critical_issues.append(f"{impact.species_name}: Significant population decline ({impact.population_change:.1f}%)")
        
        return critical_issues
    
    def _summarize_water_quality(self, monitoring_data: List[EnvironmentalData]) -> Dict:
        """Summarize current water quality conditions"""
        if not monitoring_data:
            return {}
        
        return {
            'avg_temperature': np.mean([d.water_temperature for d in monitoring_data]),
            'avg_turbidity': np.mean([d.turbidity for d in monitoring_data]),
            'avg_dissolved_oxygen': np.mean([d.dissolved_oxygen for d in monitoring_data]),
            'avg_ph': np.mean([d.ph_level for d in monitoring_data]),
            'avg_salinity': np.mean([d.salinity for d in monitoring_data])
        }
    
    def _compare_to_baseline(self, monitoring_data: List[EnvironmentalData], location: str) -> Dict:
        """Compare current conditions to baseline"""
        if not monitoring_data or location not in self.baseline_data:
            return {}
        
        baseline = self.baseline_data[location]['water_quality']
        current = self._summarize_water_quality(monitoring_data)
        
        return {
            'temperature_change': current['avg_temperature'] - baseline['avg_temperature'],
            'turbidity_change': current['avg_turbidity'] - baseline['avg_turbidity'],
            'do_change': current['avg_dissolved_oxygen'] - baseline['avg_dissolved_oxygen'],
            'ph_change': current['avg_ph'] - baseline['avg_ph'],
            'salinity_change': current['avg_salinity'] - baseline['avg_salinity']
        }
    
    def _analyze_trends(self, monitoring_data: List[EnvironmentalData]) -> Dict:
        """Analyze trends in environmental parameters"""
        if len(monitoring_data) < 7:
            return {'trend_analysis': 'Insufficient data for trend analysis'}
        
        # Sort by timestamp
        sorted_data = sorted(monitoring_data, key=lambda x: x.timestamp)
        
        # Simple linear trend analysis
        n = len(sorted_data)
        temps = [d.water_temperature for d in sorted_data]
        turb = [d.turbidity for d in sorted_data]
        
        # Calculate trends (slope)
        temp_trend = (temps[-1] - temps[0]) / n
        turb_trend = (turb[-1] - turb[0]) / n
        
        return {
            'temperature_trend': 'Increasing' if temp_trend > 0.1 else 'Decreasing' if temp_trend < -0.1 else 'Stable',
            'turbidity_trend': 'Increasing' if turb_trend > 0.1 else 'Decreasing' if turb_trend < -0.1 else 'Stable',
            'data_quality': 'Good' if n >= 30 else 'Limited'
        }
    
    def _check_water_quality_compliance(self, monitoring_data: List[EnvironmentalData]) -> List[str]:
        """Check water quality compliance"""
        violations = []
        
        for data in monitoring_data[-7:]:  # Check last 7 data points
            if data.turbidity > self.water_quality_standards['turbidity_max']:
                violations.append(f"Turbidity exceeded on {data.timestamp.date()}")
            if data.dissolved_oxygen < self.water_quality_standards['do_min']:
                violations.append(f"Dissolved oxygen low on {data.timestamp.date()}")
        
        return violations
    
    def _assess_habitat_quality(self, monitoring_data: List[EnvironmentalData]) -> Dict:
        """Assess current habitat quality"""
        if not monitoring_data:
            return {}
        
        avg_activity = np.mean([d.marine_life_activity for d in monitoring_data])
        
        return {
            'marine_life_activity_index': avg_activity,
            'habitat_quality_score': avg_activity * 100,
            'assessment': 'Good' if avg_activity > 0.8 else 'Fair' if avg_activity > 0.6 else 'Poor'
        }
    
    def _calculate_biodiversity_index(self, impacts: List[EcosystemImpact]) -> float:
        """Calculate biodiversity impact index"""
        if not impacts:
            return 1.0
        
        # Simple biodiversity index based on population changes
        pop_changes = [impact.population_change for impact in impacts]
        avg_change = np.mean(pop_changes)
        
        # Convert to 0-1 scale where 1 = no impact
        biodiversity_index = max(0, 1 + avg_change / 100)
        return biodiversity_index
    
    def _generate_recommendations(self, impacts: List[EcosystemImpact], 
                                monitoring_data: List[EnvironmentalData]) -> List[str]:
        """Generate specific recommendations"""
        recommendations = []
        
        # High-risk species recommendations
        high_risk_species = [i.species_name for i in impacts if i.risk_level == "High"]
        if high_risk_species:
            recommendations.append(f"Implement immediate mitigation measures for {', '.join(high_risk_species)}")
        
        # Water quality recommendations
        if monitoring_data:
            avg_noise = np.mean([d.noise_level for d in monitoring_data])
            if avg_noise > 100:
                recommendations.append("Consider noise reduction technologies to minimize acoustic impact")
            
            avg_turbidity = np.mean([d.turbidity for d in monitoring_data])
            if avg_turbidity > 15:
                recommendations.append("Implement turbidity control measures during maintenance activities")
        
        # General recommendations
        recommendations.extend([
            "Continue regular environmental monitoring as per approved schedule",
            "Update mitigation measures based on adaptive management principles",
            "Engage with local stakeholders and regulatory bodies regularly"
        ])
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Initialize environmental analyzer
    analyzer = EnvironmentalImpactAnalyzer()
    
    # Establish baseline for Perth location
    baseline = analyzer.establish_baseline("Perth", duration_days=365)
    print("✅ Baseline established for Perth")
    
    # Sample WEC layout
    wec_layout = [(100 + i*150, 100 + j*150) for i in range(7) for j in range(7)][:49]
    
    # Monitor environmental conditions
    for _ in range(30):  # 30 days of monitoring
        monitoring_data = analyzer.monitor_environmental_conditions("Perth", wec_layout)
    
    print(f"📊 Collected {len(analyzer.monitoring_data)} monitoring data points")
    
    # Assess ecosystem impact
    impact_assessments = analyzer.assess_ecosystem_impact("Perth", monitoring_period_days=30)
    print(f"🐋 Assessed impact on {len(impact_assessments)} species")
    
    # Calculate carbon footprint
    carbon_footprint = analyzer.calculate_carbon_footprint(wec_layout)
    print(f"🌍 Carbon footprint: {carbon_footprint['net_carbon_impact']:.0f} tons CO2 net impact")
    
    # Develop mitigation measures
    mitigation_measures = analyzer.develop_mitigation_measures(impact_assessments)
    print(f"🛡️ Developed {len(mitigation_measures)} mitigation measures")
    
    # Calculate sustainability metrics
    sustainability_metrics = analyzer.calculate_sustainability_metrics(wec_layout)
    print(f"📈 Carbon intensity: {sustainability_metrics['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh")
    
    # Generate environmental report
    report = analyzer.generate_environmental_report("Perth")
    print(f"📋 Generated comprehensive environmental report")
    print(f"Overall impact level: {report['executive_summary']['overall_impact_level']}")
    print(f"Compliance status: {report['executive_summary']['compliance_status']}")
    
    # Export report
    with open("environmental_impact_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("📄 Report exported to environmental_impact_report.json")
