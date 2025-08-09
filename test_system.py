#!/usr/bin/env python3
"""
Test script for Wave Energy Farm Optimization System
Validates all components and provides usage examples
"""

import sys
import numpy as np
import traceback

def test_economic_model():
    """Test the economic analysis module"""
    print("🧪 Testing Economic Model...")
    
    try:
        from economic_model import WECEconomicModel, EconomicWECOptimizer
        
        # Initialize economic model
        economic_model = WECEconomicModel()
        
        # Test with sample data
        np.random.seed(42)
        sample_coords = np.random.uniform(0, 1000, 49 * 2)
        sample_power = 4e6  # 4 MW
        
        # Test economic analysis
        summary, economics = economic_model.get_economic_summary(sample_coords, 49, sample_power)
        
        print(f"  ✅ NPV: ${economics['npv']:,.0f}")
        print(f"  ✅ ROI: {economics['roi_percent']:.1f}%")
        print(f"  ✅ Payback: {economics['payback_period_years']:.1f} years")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_config_management():
    """Test the configuration management system"""
    print("\n🧪 Testing Configuration Management...")
    
    try:
        from config import ConfigManager, config
        
        # Test default configuration
        config.print_config_summary()
        
        # Test parameter updates
        original_price = config.economics.electricity_price
        config.update_economic_config(electricity_price=0.15)
        
        if config.economics.electricity_price == 0.15:
            print("  ✅ Configuration update successful")
        else:
            print("  ❌ Configuration update failed")
            return False
        
        # Restore original value
        config.update_economic_config(electricity_price=original_price)
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_api_components():
    """Test API components (without starting server)"""
    print("\n🧪 Testing API Components...")
    
    try:
        from api import WECOptimizationAPI, generate_random_layout
        
        # Test API class
        api = WECOptimizationAPI()
        
        # Test coordinate validation
        test_coords = [100, 100, 200, 200, 300, 300]
        valid, message = api.validate_coordinates(test_coords, 3)
        
        if valid:
            print("  ✅ Coordinate validation working")
        else:
            print(f"  ❌ Coordinate validation failed: {message}")
            return False
        
        # Test power prediction
        power = api.predict_power(test_coords, 3)
        if power > 0:
            print(f"  ✅ Power prediction: {power/1e6:.2f} MW")
        else:
            print("  ❌ Power prediction failed")
            return False
        
        # Test layout generation
        layout = generate_random_layout(10, {'x_max': 1000, 'y_max': 1000, 'min_distance': 50})
        if len(layout) == 20:  # 10 WECs * 2 coordinates
            print("  ✅ Layout generation working")
        else:
            print("  ❌ Layout generation failed")
            return False
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_data_availability():
    """Test if required data files are available"""
    print("\n🧪 Testing Data Availability...")
    
    import os
    
    required_files = [
        'WEC_Perth_100.csv',
        'WEC_Perth_49.csv', 
        'WEC_Sydney_100.csv',
        'WEC_Sydney_49.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"  ❌ Missing data files: {missing_files}")
        return False
    else:
        print("  ✅ All data files present")
        return True

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\n🧪 Testing Dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib',
        'plotly', 'streamlit', 'xgboost', 'shap'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n  📦 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("  ✅ All dependencies available")
        return True

def run_comprehensive_test():
    """Run comprehensive system test"""
    print("🚀 Wave Energy Farm Optimization System Test")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Files", test_data_availability),
        ("Economic Model", test_economic_model),
        ("Configuration", test_config_management),
        ("API Components", test_api_components),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"  🔴 {test_name} test failed")
        except Exception as e:
            print(f"  🔴 {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 Next steps:")
        print("  1. Run Streamlit app: streamlit run streamlit_app.py")
        print("  2. Start API server: python api.py")
        print("  3. Open Jupyter notebook: jupyter notebook")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False
    
    return True

def show_usage_examples():
    """Show usage examples for different components"""
    print("\n📚 Usage Examples:")
    print("=" * 40)
    
    print("\n1. 💰 Economic Analysis:")
    print("""
from economic_model import WECEconomicModel

economic_model = WECEconomicModel()
coordinates = np.random.uniform(0, 1000, 98)  # 49 WECs
power_output = 4e6  # 4 MW

summary, economics = economic_model.get_economic_summary(coordinates, 49, power_output)
print(f"NPV: ${economics['npv']:,.0f}")
    """)
    
    print("\n2. ⚙️ Configuration Management:")
    print("""
from config import config

# Update economic parameters
config.update_economic_config(
    electricity_price=0.15,
    project_lifetime=30
)

# Print current configuration
config.print_config_summary()
    """)
    
    print("\n3. 🌐 API Usage (cURL examples):")
    print("""
# Health check
curl http://localhost:5000/api/health

# Predict power output
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"coordinates": [100,100,200,200,300,300], "n_wecs": 3}'

# Economic analysis
curl -X POST http://localhost:5000/api/economics \\
  -H "Content-Type: application/json" \\
  -d '{"coordinates": [100,100,200,200], "n_wecs": 2, "power_output": 200000}'
    """)

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success and len(sys.argv) > 1 and sys.argv[1] == '--examples':
        show_usage_examples()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
