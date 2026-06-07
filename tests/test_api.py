import os
import sys
import io
import base64
import asyncio
import numpy as np
from PIL import Image
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import generate_test_image, save_image_file


BASE_URL = "http://localhost:8000"


def create_test_image(size=(256, 256)) -> bytes:
    img = generate_test_image(size)
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format='PNG')
    return buffer.getvalue()


def test_health_endpoint():
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✓ Health endpoint OK")
        return True
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
        return False


def test_status_endpoint():
    print("\n=== Testing Status Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Device: {data['system']['device']}")
        print(f"Model loaded: {data['system']['model_loaded']}")
        print(f"Queue: {data['queue']}")
        assert response.status_code == 200
        print("✓ Status endpoint OK")
        return True
    except Exception as e:
        print(f"✗ Status endpoint failed: {e}")
        return False


def test_single_image_super_resolve():
    print("\n=== Testing Single Image Super-Resolve ===")
    try:
        test_img_bytes = create_test_image((256, 256))
        
        files = {"file": ("test.png", test_img_bytes, "image/png")}
        params = {"scale": 2, "output_format": "PNG"}
        
        response = requests.post(
            f"{BASE_URL}/super-resolve",
            files=files,
            params=params
        )
        
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Processing-Time: {response.headers.get('X-Processing-Time-MS')} ms")
        print(f"Memory-Usage: {response.headers.get('X-Memory-Usage-MB')} MB")
        print(f"Blur-Type: {response.headers.get('X-Blur-Type')}")
        print(f"Blur-Confidence: {response.headers.get('X-Blur-Confidence')}")
        
        assert response.status_code == 200
        assert "image/" in response.headers.get('Content-Type', '')
        
        output_img = Image.open(io.BytesIO(response.content))
        print(f"Input size: 256x256, Output size: {output_img.size}")
        assert output_img.size == (512, 512)
        
        os.makedirs("test_outputs", exist_ok=True)
        output_img.save("test_outputs/single_output.png")
        print("✓ Single image super-resolve OK")
        return True
    except Exception as e:
        print(f"✗ Single image super-resolve failed: {e}")
        return False


def test_json_response():
    print("\n=== Testing JSON Response Endpoint ===")
    try:
        test_img_bytes = create_test_image((128, 128))
        
        files = {"file": ("test.png", test_img_bytes, "image/png")}
        params = {"scale": 4}
        
        response = requests.post(
            f"{BASE_URL}/super-resolve/json",
            files=files,
            params=params
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        
        assert response.status_code == 200
        assert data["success"] == True
        print(f"Scale: {data['scale']}")
        print(f"Metrics: {data['metrics']}")
        print(f"Blur Analysis: {data['blur_analysis']['blur_type']} "
              f"(confidence: {data['blur_analysis']['confidence']})")
        
        img_data = base64.b64decode(data["output_image"].split(",")[1])
        output_img = Image.open(io.BytesIO(img_data))
        print(f"Output size: {output_img.size}")
        assert output_img.size == (512, 512)
        
        output_img.save("test_outputs/json_output.png")
        print("✓ JSON response OK")
        return True
    except Exception as e:
        print(f"✗ JSON response failed: {e}")
        return False


def test_blur_analysis():
    print("\n=== Testing Blur Analysis ===")
    try:
        test_img_bytes = create_test_image((256, 256))
        
        files = {"file": ("test.png", test_img_bytes, "image/png")}
        
        response = requests.post(
            f"{BASE_URL}/analyze-blur",
            files=files
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        
        assert response.status_code == 200
        assert data["success"] == True
        print(f"Blur Type: {data['blur_analysis']['blur_type']}")
        print(f"Confidence: {data['blur_analysis']['confidence']}")
        print(f"Severity: {data['blur_analysis']['severity']}")
        print("✓ Blur analysis OK")
        return True
    except Exception as e:
        print(f"✗ Blur analysis failed: {e}")
        return False


def test_batch_processing():
    print("\n=== Testing Batch Processing ===")
    try:
        files = []
        for i in range(3):
            img_bytes = create_test_image((128, 128))
            files.append(("files", (f"test_{i}.png", img_bytes, "image/png")))
        
        params = {"scale": 2, "wait": "true"}
        
        response = requests.post(
            f"{BASE_URL}/batch/super-resolve",
            files=files,
            params=params
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        
        assert response.status_code == 200
        assert data["success"] == True
        print(f"Request ID: {data['request_id']}")
        print(f"Batch size: {data['batch_size']}")
        print(f"Total processing time: {data['total_processing_time_ms']} ms")
        
        assert len(data["results"]) == 3
        
        os.makedirs("test_outputs", exist_ok=True)
        for i, result in enumerate(data["results"]):
            img_data = base64.b64decode(result["image_data"].split(",")[1])
            output_img = Image.open(io.BytesIO(img_data))
            output_img.save(f"test_outputs/batch_output_{i}.png")
            print(f"  Result {i}: {result['filename']} - {output_img.size}")
        
        print("✓ Batch processing OK")
        return True
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        return False


def test_queue_stats():
    print("\n=== Testing Queue Stats ===")
    try:
        response = requests.get(f"{BASE_URL}/queue/stats")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✓ Queue stats OK")
        return True
    except Exception as e:
        print(f"✗ Queue stats failed: {e}")
        return False


def test_large_image():
    print("\n=== Testing Large Image (1024x1024) ===")
    try:
        test_img_bytes = create_test_image((1024, 1024))
        
        files = {"file": ("large_test.png", test_img_bytes, "image/png")}
        params = {"scale": 2}
        
        response = requests.post(
            f"{BASE_URL}/super-resolve",
            files=files,
            params=params,
            timeout=120
        )
        
        print(f"Status: {response.status_code}")
        print(f"Processing-Time: {response.headers.get('X-Processing-Time-MS')} ms")
        print(f"Memory-Usage: {response.headers.get('X-Memory-Usage-MB')} MB")
        
        assert response.status_code == 200
        
        output_img = Image.open(io.BytesIO(response.content))
        print(f"Input size: 1024x1024, Output size: {output_img.size}")
        assert output_img.size == (2048, 2048)
        
        output_img.save("test_outputs/large_output.png")
        print("✓ Large image processing OK")
        return True
    except Exception as e:
        print(f"✗ Large image processing failed: {e}")
        return False


def test_batch_error_handling():
    print("\n=== Testing Batch Error Handling (207 Multi-Status) ===")
    try:
        files = []
        
        img_bytes1 = create_test_image((128, 128))
        files.append(("files", ("good_1.png", img_bytes1, "image/png")))
        
        corrupt_bytes = b"This is not a valid image file"
        files.append(("files", ("corrupt.png", corrupt_bytes, "image/png")))
        
        img_bytes2 = create_test_image((64, 64))
        files.append(("files", ("good_2.png", img_bytes2, "image/png")))
        
        params = {"scale": 4, "wait": "true"}
        
        response = requests.post(
            f"{BASE_URL}/batch/super-resolve",
            files=files,
            params=params
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 207, f"Expected 207 Multi-Status, got {response.status_code}"
        
        data = response.json()
        
        print(f"Success: {data['success']}")
        print(f"Batch size: {data['batch_size']}")
        print(f"Success count: {data['success_count']}")
        print(f"Failed count: {data['failed_count']}")
        
        assert data['success'] == False
        assert data['batch_size'] == 3
        assert data['success_count'] == 2
        assert data['failed_count'] == 1
        
        for i, result in enumerate(data['results']):
            print(f"  Item {i}: {result['filename']} - status={result['status']}")
            if result['status'] == 'failed':
                print(f"    Error: {result['error']}")
                assert result['error'] is not None
                assert result['image_data'] is None
                assert result['metrics'] is None
                assert result['blur_analysis'] is None
            else:
                assert result['error'] is None
                assert result['image_data'] is not None
                assert result['metrics'] is not None
                assert result['blur_analysis'] is not None
        
        assert data['results'][0]['status'] == 'success'
        assert data['results'][1]['status'] == 'failed'
        assert data['results'][2]['status'] == 'success'
        
        print("✓ Batch error handling (207 Multi-Status) OK")
        return True
    except Exception as e:
        print(f"✗ Batch error handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_all_good_returns_200():
    print("\n=== Testing Batch All Good Returns 200 ===")
    try:
        files = []
        for i in range(2):
            img_bytes = create_test_image((128, 128))
            files.append(("files", (f"good_{i}.png", img_bytes, "image/png")))
        
        params = {"scale": 2, "wait": "true"}
        
        response = requests.post(
            f"{BASE_URL}/batch/super-resolve",
            files=files,
            params=params
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] == True
        assert data['success_count'] == 2
        assert data['failed_count'] == 0
        
        for result in data['results']:
            assert result['status'] == 'success'
            assert result['error'] is None
        
        print("✓ Batch all good returns 200 OK")
        return True
    except Exception as e:
        print(f"✗ Batch all good returns 200 failed: {e}")
        return False


def test_different_blur_types():
    print("\n=== Testing Different Blur Types ===")
    try:
        import cv2
        
        original = generate_test_image((256, 256))
        
        gaussian_blur = cv2.GaussianBlur(original, (15, 15), 5)
        motion_blur = cv2.filter2D(original, -1, np.ones((15, 1)) / 15)
        downsample = cv2.resize(cv2.resize(original, (64, 64)), (256, 256), interpolation=cv2.INTER_NEAREST)
        
        blur_types = {
            "original": original,
            "gaussian": gaussian_blur,
            "motion": motion_blur,
            "downsample": downsample
        }
        
        for name, img in blur_types.items():
            buffer = io.BytesIO()
            Image.fromarray(img).save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            
            files = {"file": (f"{name}.png", img_bytes, "image/png")}
            
            response = requests.post(
                f"{BASE_URL}/analyze-blur",
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  {name:12s} -> {data['blur_analysis']['blur_type']:12s} "
                      f"(conf: {data['blur_analysis']['confidence']:.3f})")
        
        print("✓ Blur type classification test complete")
        return True
    except Exception as e:
        print(f"✗ Blur type classification test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Blind Super-Resolution API Test Suite")
    print("=" * 60)
    
    tests = [
        test_health_endpoint,
        test_status_endpoint,
        test_queue_stats,
        test_blur_analysis,
        test_single_image_super_resolve,
        test_json_response,
        test_batch_processing,
        test_batch_error_handling,
        test_batch_all_good_returns_200,
        test_large_image,
        test_different_blur_types
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
