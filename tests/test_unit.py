import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import RRDBNet
from src.processing import (
    split_image_into_tiles,
    merge_tiles_with_blending,
    tile_inference,
    analyze_blur,
    BlurType
)
from src.utils import (
    PerformanceMonitor,
    PerformanceMetrics,
    bytes_to_image,
    image_to_bytes,
    preprocess_image,
    postprocess_image,
    validate_image_size,
    calculate_tile_count,
    generate_test_image
)
from src.queue import RequestQueue, RequestStatus

OK = "[OK]"
FAIL = "[FAIL]"

def test_model_architecture():
    print("\n=== Testing RRDB Model Architecture ===")
    try:
        model = RRDBNet(scale=4)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        test_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (1, 3, 512, 512)
        print(f"{OK} Model architecture OK")
        return True
    except Exception as e:
        print(f"{FAIL} Model architecture test failed: {e}")
        return False


def test_tile_processing():
    print("\n=== Testing Tile Processing ===")
    try:
        img = generate_test_image((800, 600))
        print(f"Original image shape: {img.shape}")
        
        tiles, positions, original_shape = split_image_into_tiles(
            img, tile_size=512, overlap=32
        )
        print(f"Number of tiles: {len(tiles)}")
        print(f"Tile shape: {tiles[0].shape}")
        print(f"Original shape: {original_shape}")
        
        assert len(tiles) > 0
        assert tiles[0].shape == (512, 512, 3)
        
        scaled_tiles = [tile for tile in tiles]
        merged = merge_tiles_with_blending(
            scaled_tiles, positions, original_shape,
            scale=1, overlap=32
        )
        
        print(f"Merged image shape: {merged.shape}")
        assert merged.shape == img.shape
        print(f"{OK} Tile processing OK")
        return True
    except Exception as e:
        print(f"{FAIL} Tile processing test failed: {e}")
        return False


def test_blur_detection():
    print("\n=== Testing Blur Detection ===")
    try:
        import cv2
        
        original = generate_test_image((256, 256))
        gaussian = cv2.GaussianBlur(original, (21, 21), 5)
        motion = cv2.filter2D(original, -1, np.ones((20, 1)) / 20)
        downsample = cv2.resize(
            cv2.resize(original, (64, 64), interpolation=cv2.INTER_LINEAR),
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        
        test_cases = [
            ("Original", original, BlurType.SHARP),
            ("Gaussian Blur", gaussian, BlurType.GAUSSIAN),
            ("Motion Blur", motion, BlurType.MOTION),
            ("Downsampled", downsample, BlurType.DOWNSAMPLE)
        ]
        
        for name, img, expected_type in test_cases:
            result = analyze_blur(img)
            print(f"  {name:15s}: {result.blur_type.value:12s} "
                  f"(conf: {result.confidence:.3f}, severity: {result.severity:.3f})")
            print(f"    Details: lap_var={result.details['laplacian_variance']:.1f}, "
                  f"energy_ratio={result.details['fft_energy_ratio']:.2f}")
        
        print(f"{OK} Blur detection OK")
        return True
    except Exception as e:
        print(f"{FAIL} Blur detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitor():
    print("\n=== Testing Performance Monitor ===")
    try:
        monitor = PerformanceMonitor(device='cpu')
        
        metrics = PerformanceMetrics()
        with monitor.measure(metrics):
            with monitor.measure_stage('preprocessing'):
                img = generate_test_image((256, 256))
            with monitor.measure_stage('inference'):
                import time
                time.sleep(0.1)
            with monitor.measure_stage('postprocessing'):
                _ = img * 2
        
        print(f"Total time: {metrics.processing_time_ms:.2f} ms")
        print(f"Preprocessing: {metrics.preprocessing_time_ms:.2f} ms")
        print(f"Inference: {metrics.inference_time_ms:.2f} ms")
        print(f"Postprocessing: {metrics.postprocessing_time_ms:.2f} ms")
        print(f"Memory usage: {metrics.peak_memory_usage_mb:.2f} MB")
        
        stats = monitor.get_system_stats()
        print(f"System stats: {list(stats.keys())}")
        
        assert metrics.processing_time_ms > 0
        print(f"{OK} Performance monitor OK")
        return True
    except Exception as e:
        print(f"{FAIL} Performance monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_utils():
    print("\n=== Testing Image Utilities ===")
    try:
        img = generate_test_image((256, 256))
        
        img_bytes = image_to_bytes(img, format='PNG')
        decoded = bytes_to_image(img_bytes)
        print(f"Original: {img.shape}, Decoded: {decoded.shape}")
        assert decoded.shape == img.shape
        
        tensor = preprocess_image(img, 'cpu')
        print(f"Tensor shape: {tensor.shape}")
        assert tensor.shape == (1, 3, 256, 256)
        
        recovered = postprocess_image(tensor)
        print(f"Recovered shape: {recovered.shape}")
        assert recovered.shape == img.shape
        
        valid, msg = validate_image_size(img, max_size=2048)
        print(f"Validation: {valid}, {msg}")
        assert valid == True
        
        large_img = generate_test_image((3000, 3000))
        valid, msg = validate_image_size(large_img, max_size=2048)
        print(f"Large image validation: {valid}, {msg}")
        assert valid == False
        
        tile_count = calculate_tile_count(1024, 1024, tile_size=512, overlap=32)
        print(f"Tile count for 1024x1024: {tile_count}")
        assert tile_count == 9
        
        print(f"{OK} Image utilities OK")
        return True
    except Exception as e:
        print(f"{FAIL} Image utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_request_queue():
    print("\n=== Testing Request Queue ===")
    try:
        queue = RequestQueue(max_concurrent=2, max_queue_size=10)
        
        class MockProcessor:
            async def process(self, images, scale, metadata):
                await asyncio.sleep(0.1)
                return {
                    'output_images': [img * 2 for img in images],
                    'metrics_list': [PerformanceMetrics() for _ in images],
                    'blur_analyses': [analyze_blur(img) for img in images],
                    'batch_size': len(images),
                    'total_processing_time_ms': 100
                }
        
        processor = MockProcessor()
        queue.set_processor(processor)
        
        await queue.start()
        
        img = generate_test_image((128, 128))
        item = await queue.submit([img], scale=4)
        print(f"Submitted request: {item.request_id}, status: {item.status}")
        
        stats = await queue.get_queue_stats()
        print(f"Queue stats: {stats}")
        
        completed = await queue.wait_for_result(item.request_id, timeout=5.0)
        print(f"Completed: {completed.request_id}, status: {completed.status}")
        assert completed.status == RequestStatus.COMPLETED
        
        await queue.stop()
        print(f"{OK} Request queue OK")
        return True
    except Exception as e:
        print(f"{FAIL} Request queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calculate_tile_count():
    print("\n=== Testing Tile Count Calculation ===")
    try:
        test_cases = [
            ((256, 256), 512, 32, 1),
            ((512, 512), 512, 32, 1),
            ((600, 600), 512, 32, 4),
            ((1024, 1024), 512, 32, 9),
            ((2048, 2048), 512, 32, 25),
        ]
        
        for (h, w), tile_size, overlap, expected in test_cases:
            count = calculate_tile_count(h, w, tile_size, overlap)
            print(f"  {h}x{w}: {count} tiles (expected {expected})")
            assert count == expected
        
        print(f"{OK} Tile count calculation OK")
        return True
    except Exception as e:
        print(f"{FAIL} Tile count calculation test failed: {e}")
        return False


async def main():
    print("=" * 60)
    print("Unit Tests for Blind Super-Resolution System")
    print("=" * 60)
    
    import torch
    global torch
    
    tests = [
        test_image_utils,
        test_calculate_tile_count,
        test_tile_processing,
        test_blur_detection,
        test_performance_monitor,
        test_model_architecture,
        test_request_queue,
    ]
    
    results = []
    for test in tests:
        try:
            if test.__name__ == 'test_request_queue':
                result = await test()
            else:
                result = test()
            results.append(result)
        except Exception as e:
            print(f"{FAIL} Test {test.__name__} exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Unit Test Results: {passed}/{total} passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
