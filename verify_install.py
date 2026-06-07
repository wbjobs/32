import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def verify_installation():
    OK = "[OK]"
    FAIL = "[FAIL]"
    WARN = "[WARN]"
    
    print("=" * 60)
    print("Installation Verification")
    print("=" * 60)
    
    checks = []
    
    print("\n1. Checking Python version...")
    print(f"   Python {sys.version}")
    checks.append(("Python 3.8+", sys.version_info >= (3, 8)))
    
    print("\n2. Checking required packages...")
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("psutil", "psutil"),
    ]
    
    for pkg_name, display_name in packages:
        try:
            __import__(pkg_name)
            print(f"   {OK} {display_name} installed")
            checks.append((display_name, True))
        except ImportError as e:
            print(f"   {FAIL} {display_name} not found: {e}")
            checks.append((display_name, False))
    
    print("\n3. Checking project structure...")
    project_files = [
        "main.py",
        "config.py",
        "requirements.txt",
        "src/models/rrdb.py",
        "src/processing/tile_inference.py",
        "src/processing/blur_detection.py",
        "src/queue/request_queue.py",
        "src/utils/performance_monitor.py",
        "src/utils/image_utils.py",
        "src/core/super_resolution.py",
    ]
    
    for file_path in project_files:
        if os.path.exists(file_path):
            print(f"   {OK} {file_path}")
            checks.append((file_path, True))
        else:
            print(f"   {FAIL} {file_path} - MISSING")
            checks.append((file_path, False))
    
    print("\n4. Checking PyTorch CUDA availability...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"   {OK} CUDA available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print(f"   {WARN} CUDA not available (will use CPU fallback)")
        checks.append(("PyTorch", True))
    except ImportError:
        print(f"   {FAIL} PyTorch not installed")
        checks.append(("PyTorch", False))
        cuda_available = False
    
    print("\n5. Checking model directory...")
    os.makedirs("models", exist_ok=True)
    model_files = os.listdir("models")
    if model_files:
        print(f"   {OK} Model directory contains: {model_files}")
    else:
        print(f"   {WARN} Model directory is empty.")
        print("        Please place your pre-trained RRDB model in the 'models' directory.")
        print("        Expected model: models/RRDB_ESRGAN_x4.pth")
    checks.append(("Model directory", os.path.isdir("models")))
    
    print("\n6. Testing module imports...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.models import RRDBNet
        from src.processing import analyze_blur, tile_inference
        from src.utils import PerformanceMonitor, bytes_to_image
        from src.core import SuperResolutionProcessor
        from src.queue import RequestQueue
        
        print(f"   {OK} All modules imported successfully")
        checks.append(("Module imports", True))
    except Exception as e:
        print(f"   {FAIL} Import error: {e}")
        import traceback
        traceback.print_exc()
        checks.append(("Module imports", False))
    
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"Verification Results: {passed}/{total} checks passed")
    print("=" * 60)
    
    if passed == total:
        print("\n[SUCCESS] All checks passed! You can start the server with:")
        print("   python main.py")
        print("   or")
        print("   start_server.bat")
        return True
    else:
        failed = [name for name, ok in checks if not ok]
        print(f"\n[FAILED] Failed checks: {failed}")
        print("\nPlease install missing dependencies with:")
        print("   pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
