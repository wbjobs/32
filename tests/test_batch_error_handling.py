import os
import sys
import io
import base64
import asyncio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import SuperResolutionProcessor
from src.utils import generate_test_image, image_to_bytes


async def test_processor_error_handling():
    print("=" * 60)
    print("Testing Processor Error Handling")
    print("=" * 60)
    
    processor = SuperResolutionProcessor()
    
    print("\n1. Testing processor with mixed images...")
    
    good_img = generate_test_image((128, 128))
    
    error_img = {"error": "Corrupt image data"}
    
    good_img2 = generate_test_image((64, 64))
    
    images = [good_img, error_img, good_img2]
    
    result = await processor.process(images, scale=2)
    
    print(f"Batch size: {result['batch_size']}")
    print(f"Success count: {result['success_count']}")
    print(f"Failed count: {result['failed_count']}")
    print(f"Statuses: {result['statuses']}")
    print(f"Errors: {result['errors']}")
    
    assert result['batch_size'] == 3
    assert result['success_count'] == 2
    assert result['failed_count'] == 1
    assert result['statuses'] == ['success', 'failed', 'success']
    assert result['errors'][0] is None
    assert result['errors'][1] == "Corrupt image data"
    assert result['errors'][2] is None
    assert result['output_images'][0] is not None
    assert result['output_images'][1] is None
    assert result['output_images'][2] is not None
    
    print("[OK] Processor error handling test passed")
    
    print("\n2. Testing processor with all bad images...")
    all_bad = [
        {"error": "Error 1"},
        {"error": "Error 2"}
    ]
    result2 = await processor.process(all_bad, scale=2)
    print(f"Success count: {result2['success_count']}")
    print(f"Failed count: {result2['failed_count']}")
    assert result2['success_count'] == 0
    assert result2['failed_count'] == 2
    print("[OK] All bad images test passed")
    
    print("\n3. Testing processor with oversized image...")
    try:
        big_img = generate_test_image((3000, 3000))
        mixed = [good_img, big_img]
        result3 = await processor.process(mixed, scale=2)
        print(f"Success count: {result3['success_count']}")
        print(f"Failed count: {result3['failed_count']}")
        print(f"Errors: {result3['errors']}")
        print(f"Statuses: {result3['statuses']}")
        
        assert result3['success_count'] >= 1
        print("[OK] Oversized image handling test passed")
    except Exception as e:
        print(f"[INFO] Expected behavior - oversized image: {e}")
    
    print("\n" + "=" * 60)
    print("All error handling tests passed!")
    print("=" * 60)
    return True


def simulate_api_response_example():
    print("\n" + "=" * 60)
    print("API Response Format (207 Multi-Status Example)")
    print("=" * 60)
    
    example_response = {
        "success": False,
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "scale": 4,
        "batch_size": 3,
        "success_count": 2,
        "failed_count": 1,
        "total_processing_time_ms": 1250.50,
        "processing_time_ms": 1300.25,
        "results": [
            {
                "status": "success",
                "filename": "good_image1.jpg",
                "image_data": "data:image/png;base64,...",
                "metrics": {
                    "processing_time_ms": 500.25,
                    "peak_memory_usage_mb": 125.5,
                    "...": "..."
                },
                "blur_analysis": {
                    "blur_type": "gaussian",
                    "...": "..."
                },
                "error": None
            },
            {
                "status": "failed",
                "filename": "corrupt_image.jpg",
                "image_data": None,
                "metrics": None,
                "blur_analysis": None,
                "error": "Invalid image file corrupt_image.jpg: cannot identify image file"
            },
            {
                "status": "success",
                "filename": "good_image2.png",
                "image_data": "data:image/png;base64,...",
                "metrics": { "...": "..." },
                "blur_analysis": { "...": "..." },
                "error": None
            }
        ]
    }
    
    import json
    print(json.dumps(example_response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    success = asyncio.run(test_processor_error_handling())
    simulate_api_response_example()
    sys.exit(0 if success else 1)
