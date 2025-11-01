#!/usr/bin/env python3
"""
Test that all renaming worked correctly
"""

def test_imports():
    try:
        from vanasight import VanaSight
        print("✅ VanaSight import successful")
        
        from vanasight.pipeline import main
        print("✅ Pipeline import successful")
        
        pipeline = VanaSight()
        print("✅ Pipeline instantiation successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_cli():
    import subprocess
    try:
        result = subprocess.run(['vanasight', '--help'], capture_output=True, text=True)
        if 'VanaSight' in result.stdout:
            print("✅ CLI command working")
            return True
        else:
            print("❌ CLI command not found")
            return False
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing VanaSight renaming...")
    imports_ok = test_imports()
    cli_ok = test_cli()
    
    if imports_ok and cli_ok:
        print("\n🎉 All tests passed! VanaSight is ready!")
    else:
        print("\n❌ Some tests failed. Check the renaming.")
