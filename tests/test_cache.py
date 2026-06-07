import os
import sys
import io
import time
import asyncio
import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
cache_module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'cache', 'lru_cache.py')
spec = importlib.util.spec_from_file_location("lru_cache", cache_module_path)
lru_cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lru_cache_module)
LRURedisCache = lru_cache_module.LRURedisCache
CacheEntry = lru_cache_module.CacheEntry

from src.utils import generate_test_image, image_to_bytes, serialize_metrics, serialize_blur_analysis
from src.processing import analyze_blur
from src.utils.performance_monitor import PerformanceMetrics


def generate_different_test_image(size=(256, 256), pattern_id=0):
    """生成不同模式的测试图像，确保感知哈希不同"""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    if pattern_id == 0:
        # 模式1：水平渐变
        for i in range(w):
            img[:, i] = [int(i * 255 / w), int((w - i) * 255 / w), 128]
    elif pattern_id == 1:
        # 模式2：垂直渐变
        for i in range(h):
            img[i, :] = [128, int(i * 255 / h), int((h - i) * 255 / h)]
    elif pattern_id == 2:
        # 模式3：棋盘格
        for i in range(h):
            for j in range(w):
                if ((i // 32) + (j // 32)) % 2 == 0:
                    img[i, j] = [255, 255, 255]
                else:
                    img[i, j] = [0, 0, 0]
    elif pattern_id == 3:
        # 模式4：随机噪点（固定种子确保可重复）
        rng = np.random.RandomState(42 + pattern_id)
        img = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        # 模式5：同心圆
        center_h, center_w = h // 2, w // 2
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center_h) ** 2 + (j - center_w) ** 2)
                val = int(255 * np.sin(dist / 10))
                img[i, j] = [
                    np.uint8(np.clip(val, 0, 255)),
                    np.uint8(np.clip(255 - val, 0, 255)),
                    np.uint8(np.clip(abs(128 - val), 0, 255))
                ]
    
    return img


def test_image_hashing():
    print("\n=== Testing Image Hashing ===")
    try:
        import importlib.util
        hashing_module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'utils', 'image_hashing.py')
        spec = importlib.util.spec_from_file_location("image_hashing", hashing_module_path)
        hashing_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hashing_module)
        ImageHasher = hashing_module.ImageHasher
        compute_image_hash = hashing_module.compute_image_hash
        compute_hash_fast = hashing_module.compute_hash_fast
        
        hasher = ImageHasher(hash_size=8)
        
        img1 = generate_different_test_image((256, 256), pattern_id=0)
        img2 = generate_different_test_image((256, 256), pattern_id=1)
        img3 = generate_different_test_image((512, 512), pattern_id=0)
        
        key1 = hasher.generate_cache_key(img1, scale=4)
        key2 = hasher.generate_cache_key(img2, scale=4)
        key3 = hasher.generate_cache_key(img3, scale=4)
        
        print(f"Key1: {key1[:50]}...")
        print(f"Key2: {key2[:50]}...")
        print(f"Key3: {key3[:50]}...")
        
        assert key1 != key2, "Different images should have different keys"
        assert key1 != key3, "Different size images should have different keys"
        
        key1_same = hasher.generate_cache_key(img1, scale=4)
        assert key1 == key1_same, "Same image should have same key"
        
        key1_scale2 = hasher.generate_cache_key(img1, scale=2)
        assert key1 != key1_scale2, "Different scale should have different keys"
        
        fast_key = compute_hash_fast(img1, scale=4)
        print(f"Fast hash: {fast_key}")
        
        similar = hasher.is_similar(img1, img1, threshold=1)
        print(f"Self similarity: {similar}")
        assert similar == True
        
        print("[OK] Image hashing test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Image hashing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lru_cache_operations():
    print("\n=== Testing LRU Cache Operations ===")
    try:
        cache = LRURedisCache(
            max_size=3,
            ttl_seconds=3600,
            redis_enabled=False,
            key_prefix="test_cache"
        )
        
        img1 = generate_different_test_image((128, 128), pattern_id=0)
        img2 = generate_different_test_image((128, 128), pattern_id=1)
        img3 = generate_different_test_image((128, 128), pattern_id=2)
        img4 = generate_different_test_image((128, 128), pattern_id=3)
        
        key1 = cache.compute_key(img1, scale=2)
        key2 = cache.compute_key(img2, scale=2)
        key3 = cache.compute_key(img3, scale=2)
        key4 = cache.compute_key(img4, scale=2)
        
        blur1 = analyze_blur(img1)
        blur2 = analyze_blur(img2)
        
        metrics = PerformanceMetrics()
        metrics.processing_time_ms = 100.5
        metrics.peak_memory_usage_mb = 50.2
        metrics.input_size = (128, 128)
        metrics.output_size = (256, 256)
        metrics_dict = serialize_metrics(metrics)
        blur_dict1 = serialize_blur_analysis(blur1)
        blur_dict2 = serialize_blur_analysis(blur2)
        
        cache.set(key1, img1, metrics_dict, blur_dict1, scale=2, blur_type=blur1.blur_type.value)
        cache.set(key2, img2, metrics_dict, blur_dict2, scale=2, blur_type=blur2.blur_type.value)
        cache.set(key3, img3, metrics_dict, blur_dict1, scale=2, blur_type=blur1.blur_type.value)
        
        stats = cache.get_stats()
        print(f"Cache stats after 3 sets: {stats}")
        assert stats['memory_cache_size'] == 3
        
        entry1 = cache.get(key1)
        assert entry1 is not None
        assert entry1.output_image is not None
        assert entry1.access_count == 2
        print(f"Access count after get: {entry1.access_count}")
        
        cache.set(key4, img4, metrics_dict, blur_dict1, scale=2, blur_type=blur1.blur_type.value)
        
        stats = cache.get_stats()
        print(f"Cache stats after 4th set: {stats}")
        assert stats['memory_cache_size'] == 3
        
        entry_evicted = cache.get(key2)
        assert entry_evicted is None, "key2 should have been evicted (LRU - least recently used)"
        print("LRU eviction works correctly")
        
        entry1_still_exists = cache.get(key1)
        assert entry1_still_exists is not None, "key1 should still exist (was accessed recently)"
        
        entry3_exists = cache.get(key3)
        assert entry3_exists is not None, "key3 should still exist"
        
        contains_key2 = cache.contains(key2)
        contains_key1 = cache.contains(key1)
        print(f"Contains key2: {contains_key2}, key1: {contains_key1}")
        assert contains_key2 == False
        assert contains_key1 == True
        
        recent = cache.get_recent(count=2)
        print(f"Recent entries: {len(recent)}")
        assert len(recent) == 2
        
        cache.delete(key2)
        assert cache.contains(key2) == False
        print("Delete works correctly")
        
        cache.clear(clear_redis=False)
        stats = cache.get_stats()
        print(f"Cache stats after clear: {stats}")
        assert stats['memory_cache_size'] == 0
        
        print("[OK] LRU cache operations test passed")
        return True
    except Exception as e:
        print(f"[FAIL] LRU cache operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_performance():
    print("\n=== Testing Cache Performance (should be <10ms) ===")
    try:
        cache = LRURedisCache(
            max_size=100,
            ttl_seconds=3600,
            redis_enabled=False,
            key_prefix="perf_test"
        )
        
        img = generate_different_test_image((256, 256), pattern_id=0)
        key = cache.compute_key(img, scale=4)
        
        blur = analyze_blur(img)
        metrics = PerformanceMetrics()
        metrics.processing_time_ms = 2000
        metrics_dict = serialize_metrics(metrics)
        blur_dict = serialize_blur_analysis(blur)
        
        cache.set(key, img, metrics_dict, blur_dict, scale=4, blur_type=blur.blur_type.value)
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            entry = cache.get(key)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            assert entry is not None
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"Cache hit times (ms): {[f'{t:.2f}' for t in times]}")
        print(f"Average: {avg_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")
        
        assert avg_time < 10, f"Average cache hit time {avg_time:.2f}ms exceeds 10ms requirement"
        
        print(f"[OK] Cache performance test passed (avg: {avg_time:.2f}ms < 10ms)")
        return True
    except Exception as e:
        print(f"[FAIL] Cache performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_processor_cache_integration():
    print("\n=== Testing Processor Cache Integration (Simulated) ===")
    try:
        cache = LRURedisCache(
            max_size=100,
            ttl_seconds=3600,
            redis_enabled=False,
            key_prefix="integration_test"
        )
        
        img = generate_different_test_image((128, 128), pattern_id=4)
        scale = 2
        
        blur = analyze_blur(img)
        metrics = PerformanceMetrics()
        metrics.processing_time_ms = 2000.0
        metrics.peak_memory_usage_mb = 100.0
        metrics.input_size = (128, 128)
        metrics.output_size = (256, 256)
        metrics_dict = serialize_metrics(metrics)
        blur_dict = serialize_blur_analysis(blur)
        
        key = cache.compute_key(img, scale)
        
        print("First request (cache miss)...")
        start = time.perf_counter()
        entry = cache.get(key)
        miss_time = (time.perf_counter() - start) * 1000
        print(f"Cache miss check time: {miss_time:.2f}ms")
        assert entry is None
        
        print("Processing (simulated 2s)...")
        time.sleep(0.002)
        output_img = img.copy()
        
        cache.set(key, output_img, metrics_dict, blur_dict, scale, blur.blur_type.value)
        
        print("Second request (cache hit)...")
        start = time.perf_counter()
        entry2 = cache.get(key)
        hit_time = (time.perf_counter() - start) * 1000
        print(f"Cache hit time: {hit_time:.4f}ms")
        
        assert entry2 is not None
        assert entry2.output_image is not None
        assert entry2.scale == scale
        assert entry2.access_count == 2  # set时初始为1，get时+1
        
        assert hit_time < 10, f"Cache hit time {hit_time:.4f}ms exceeds 10ms requirement"
        
        speedup = 2000.0 / max(hit_time, 0.001)
        print(f"Simulated speedup: {speedup:.2f}x faster (2s -> {hit_time:.4f}ms)")
        
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
        assert stats['memory_cache_size'] == 1
        
        print("[OK] Processor cache integration test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Processor cache integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_entry_serialization():
    print("\n=== Testing Cache Entry Serialization ===")
    try:
        img = generate_different_test_image((128, 128), pattern_id=0)
        blur = analyze_blur(img)
        metrics = PerformanceMetrics()
        metrics.processing_time_ms = 1500
        metrics_dict = serialize_metrics(metrics)
        blur_dict = serialize_blur_analysis(blur)
        
        entry = CacheEntry(
            cache_key="test_key_123",
            output_image=img,
            output_image_bytes=image_to_bytes(img),
            metrics=metrics_dict,
            blur_analysis=blur_dict,
            created_at=time.time(),
            access_count=5,
            scale=4,
            blur_type=blur.blur_type.value
        )
        
        entry_dict = entry.to_dict(include_image=True)
        assert 'output_image_bytes' in entry_dict
        assert entry_dict['cache_key'] == "test_key_123"
        assert entry_dict['access_count'] == 5
        
        entry2 = CacheEntry.from_dict(entry_dict)
        assert entry2.cache_key == entry.cache_key
        assert entry2.output_image is not None
        assert entry2.output_image.shape == img.shape
        assert entry2.access_count == 5
        
        np.testing.assert_array_almost_equal(entry2.output_image, img, decimal=1)
        print("Image preserved correctly after serialization")
        
        print("[OK] Cache entry serialization test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Cache entry serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("=" * 60)
    print("LRU Cache Unit Tests")
    print("=" * 60)
    
    tests = [
        test_image_hashing,
        test_lru_cache_operations,
        test_cache_performance,
        test_cache_entry_serialization,
        test_processor_cache_integration
    ]
    
    results = []
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Cache Test Results: {passed}/{total} passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
