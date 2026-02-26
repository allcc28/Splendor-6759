"""
Environment verification script for Splendor RL project.
Tests all required dependencies and GPU availability.
"""

def check_python():
    import sys
    print(f"✅ Python {sys.version}")
    print(f"   Executable: {sys.executable}")
    return True

def check_pytorch():
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA not available (will use CPU - training will be SLOW)")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch not installed: {e}")
        return False

def check_stable_baselines3():
    try:
        import stable_baselines3
        print(f"✅ Stable-Baselines3 {stable_baselines3.__version__}")
        return True
    except ImportError:
        print("❌ Stable-Baselines3 not installed")
        print("   Install: pip install stable-baselines3[extra]")
        return False

def check_gym():
    try:
        import gym
        print(f"✅ Gym {gym.__version__}")
        return True
    except ImportError:
        print("❌ Gym not installed")
        print("   Install: pip install gym")
        return False

def check_splendor_env():
    try:
        import sys
        from pathlib import Path
        
        # Add modules to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root / "modules"))
        
        from gym_splendor_code.envs.splendor import SplendorEnv
        print("✅ Splendor environment available")
        
        # Quick test
        env = SplendorEnv()
        env.reset()
        print("   ✅ Environment reset successful")
        
        return True
    except Exception as e:
        print(f"❌ Splendor environment error: {e}")
        return False

def check_tensorboard():
    try:
        import tensorboard
        print(f"✅ TensorBoard available")
        return True
    except ImportError:
        print("⚠️  TensorBoard not installed (optional for visualization)")
        print("   Install: pip install tensorboard")
        return False

def check_yaml():
    try:
        import yaml
        print("✅ PyYAML available")
        return True
    except ImportError:
        print("❌ PyYAML not installed")
        print("   Install: pip install pyyaml")
        return False

def main():
    print("=" * 60)
    print("IFT6759 Splendor RL - Environment Check")
    print("=" * 60)
    print()
    
    results = {}
    
    print("1. Python Environment:")
    results['python'] = check_python()
    print()
    
    print("2. PyTorch & GPU:")
    results['pytorch'] = check_pytorch()
    print()
    
    print("3. Reinforcement Learning:")
    results['sb3'] = check_stable_baselines3()
    results['gym'] = check_gym()
    print()
    
    print("4. Splendor Game:")
    results['splendor'] = check_splendor_env()
    print()
    
    print("5. Optional Dependencies:")
    results['tensorboard'] = check_tensorboard()
    results['yaml'] = check_yaml()
    print()
    
    print("=" * 60)
    
    critical = ['python', 'pytorch', 'sb3', 'gym', 'splendor', 'yaml']
    critical_passed = all(results.get(k, False) for k in critical)
    
    if critical_passed:
        print("🎉 All critical dependencies satisfied!")
        print("   Ready to start implementation.")
    else:
        print("⚠️  Missing critical dependencies.")
        print("   Please install missing packages before continuing.")
        
        print("\nQuick install command:")
        print("pip install torch torchvision stable-baselines3[extra] gym pyyaml tensorboard")
    
    print("=" * 60)
    
    return critical_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
