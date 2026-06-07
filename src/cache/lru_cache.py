import os
import sys
import json
import time
import base64
import pickle
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings
from src.utils.image_hashing import ImageHasher, compute_image_hash, compute_hash_fast


try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CacheEntry:
    cache_key: str
    output_image: Optional[np.ndarray] = None
    output_image_bytes: Optional[bytes] = None
    metrics: Optional[Dict[str, Any]] = None
    blur_analysis: Optional[Dict[str, Any]] = None
    created_at: float = 0.0
    access_count: int = 0
    scale: int = 2
    blur_type: str = "unknown"
    
    def to_dict(self, include_image: bool = True) -> Dict[str, Any]:
        data = {
            'cache_key': self.cache_key,
            'metrics': self.metrics,
            'blur_analysis': self.blur_analysis,
            'created_at': self.created_at,
            'access_count': self.access_count,
            'scale': self.scale,
            'blur_type': self.blur_type,
        }
        
        if include_image and self.output_image_bytes is not None:
            data['output_image_bytes'] = base64.b64encode(self.output_image_bytes).decode('utf-8')
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        entry = cls(
            cache_key=data['cache_key'],
            metrics=data.get('metrics'),
            blur_analysis=data.get('blur_analysis'),
            created_at=data.get('created_at', time.time()),
            access_count=data.get('access_count', 0),
            scale=data.get('scale', 2),
            blur_type=data.get('blur_type', 'unknown')
        )
        
        if 'output_image_bytes' in data and data['output_image_bytes']:
            try:
                entry.output_image_bytes = base64.b64decode(data['output_image_bytes'])
                img = Image.open(io.BytesIO(entry.output_image_bytes))
                entry.output_image = np.array(img)
            except Exception:
                pass
        
        return entry


class LRURedisCache:
    def __init__(
        self,
        max_size: int = None,
        ttl_seconds: int = None,
        redis_enabled: bool = None,
        redis_host: str = None,
        redis_port: int = None,
        redis_db: int = None,
        redis_password: str = None,
        key_prefix: str = None,
        hash_size: int = None
    ):
        self.max_size = max_size or settings.cache_max_size
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self.redis_enabled = redis_enabled if redis_enabled is not None else settings.redis_enabled
        self.key_prefix = key_prefix or settings.cache_key_prefix
        self.hash_size = hash_size or settings.cache_hash_size
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        self._hasher = ImageHasher(hash_size=self.hash_size)
        
        self._redis_client = None
        self._redis_connected = False
        
        if self.redis_enabled and REDIS_AVAILABLE:
            self._init_redis(
                redis_host or settings.redis_host,
                redis_port or settings.redis_port,
                redis_db or settings.redis_db,
                redis_password or settings.redis_password
            )
            self._load_hot_from_redis()
    
    def _init_redis(self, host: str, port: int, db: int, password: Optional[str]):
        try:
            self._redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
                socket_connect_timeout=settings.redis_timeout,
                socket_timeout=settings.redis_timeout
            )
            self._redis_client.ping()
            self._redis_connected = True
            print(f"[Cache] Connected to Redis at {host}:{port}/{db}")
        except Exception as e:
            print(f"[Cache] Failed to connect to Redis: {e}. Using in-memory cache only.")
            self._redis_connected = False
    
    def _load_hot_from_redis(self, count: int = 50):
        if not self._redis_connected:
            return
        
        try:
            pattern = f"{self.key_prefix}:*"
            keys = self._redis_client.keys(pattern)
            
            entries = []
            for key in keys:
                try:
                    data = self._redis_client.get(key)
                    if data:
                        entry_data = json.loads(data)
                        if 'access_count' in entry_data:
                            entries.append((entry_data['access_count'], key, entry_data))
                except Exception:
                    continue
            
            entries.sort(key=lambda x: x[0], reverse=True)
            
            for _, key, entry_data in entries[:count]:
                try:
                    short_key = key.decode('utf-8').replace(f"{self.key_prefix}:", "")
                    entry = CacheEntry.from_dict(entry_data)
                    with self._lock:
                        if len(self._cache) < self.max_size:
                            self._cache[short_key] = entry
                            self._cache.move_to_end(short_key)
                except Exception:
                    continue
            
            print(f"[Cache] Loaded {min(len(entries), count)} hot entries from Redis into memory")
        except Exception as e:
            print(f"[Cache] Failed to load hot entries from Redis: {e}")
    
    def _get_full_key(self, cache_key: str) -> str:
        return f"{self.key_prefix}:{cache_key}"
    
    def compute_key(self, img: np.ndarray, scale: int) -> str:
        return self._hasher.generate_cache_key(img, scale)
    
    def compute_key_fast(self, img: np.ndarray, scale: int) -> str:
        return compute_hash_fast(img, scale)
    
    def get(self, cache_key: str) -> Optional[CacheEntry]:
        start_time = time.time()
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                self._cache.move_to_end(cache_key)
                entry.access_count += 1
                
                self._async_update_redis_access(cache_key, entry)
                
                print(f"[Cache] Hit (memory) in {(time.time() - start_time)*1000:.2f}ms")
                return entry
        
        if self._redis_connected:
            try:
                full_key = self._get_full_key(cache_key)
                data = self._redis_client.get(full_key)
                if data:
                    entry_data = json.loads(data)
                    entry = CacheEntry.from_dict(entry_data)
                    entry.access_count = entry_data.get('access_count', 0) + 1
                    
                    with self._lock:
                        if len(self._cache) >= self.max_size:
                            self._cache.popitem(last=False)
                        self._cache[cache_key] = entry
                    
                    self._async_update_redis_access(cache_key, entry)
                    
                    print(f"[Cache] Hit (Redis) in {(time.time() - start_time)*1000:.2f}ms")
                    return entry
            except Exception as e:
                print(f"[Cache] Redis get error: {e}")
        
        print(f"[Cache] Miss in {(time.time() - start_time)*1000:.2f}ms")
        return None
    
    def _async_update_redis_access(self, cache_key: str, entry: CacheEntry):
        if not self._redis_connected:
            return
        
        try:
            full_key = self._get_full_key(cache_key)
            self._redis_client.expire(full_key, self.ttl_seconds)
            
            pipeline = self._redis_client.pipeline()
            pipeline.hincrby(full_key, 'access_count', 1)
            pipeline.expire(full_key, self.ttl_seconds)
            pipeline.execute()
        except Exception:
            pass
    
    def set(self, cache_key: str, 
            output_image: np.ndarray,
            metrics: Dict[str, Any],
            blur_analysis: Dict[str, Any],
            scale: int,
            blur_type: str = "unknown") -> CacheEntry:
        start_time = time.time()
        
        output_bytes = self._image_to_bytes(output_image)
        
        entry = CacheEntry(
            cache_key=cache_key,
            output_image=output_image,
            output_image_bytes=output_bytes,
            metrics=metrics,
            blur_analysis=blur_analysis,
            created_at=time.time(),
            access_count=1,
            scale=scale,
            blur_type=blur_type
        )
        
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
            else:
                if len(self._cache) >= self.max_size:
                    evicted_key, evicted_entry = self._cache.popitem(last=False)
                    print(f"[Cache] Evicted: {evicted_key[:20]}... (accesses: {evicted_entry.access_count})")
                self._cache[cache_key] = entry
        
        if self._redis_connected:
            self._async_write_to_redis(cache_key, entry)
        
        print(f"[Cache] Set in {(time.time() - start_time)*1000:.2f}ms, size: {self.max_size}")
        return entry
    
    def _async_write_to_redis(self, cache_key: str, entry: CacheEntry):
        if not self._redis_connected:
            return
        
        try:
            full_key = self._get_full_key(cache_key)
            entry_dict = entry.to_dict(include_image=True)
            data = json.dumps(entry_dict)
            
            self._redis_client.setex(
                full_key,
                self.ttl_seconds,
                data
            )
        except Exception as e:
            print(f"[Cache] Redis write error: {e}")
    
    def _image_to_bytes(self, img: np.ndarray, format: str = 'PNG') -> bytes:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format)
        return buffer.getvalue()
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            memory_size = len(self._cache)
            access_counts = [e.access_count for e in self._cache.values()]
            total_accesses = sum(access_counts)
            avg_access = total_accesses / memory_size if memory_size > 0 else 0
        
        redis_keys = 0
        if self._redis_connected:
            try:
                pattern = f"{self.key_prefix}:*"
                redis_keys = len(self._redis_client.keys(pattern))
            except Exception:
                pass
        
        return {
            "memory_cache_size": memory_size,
            "memory_cache_max": self.max_size,
            "redis_connected": self._redis_connected,
            "redis_cache_size": redis_keys,
            "total_accesses_tracked": total_accesses,
            "avg_access_per_entry": round(avg_access, 2),
            "ttl_seconds": self.ttl_seconds
        }
    
    def clear(self, clear_redis: bool = True):
        with self._lock:
            self._cache.clear()
        
        if clear_redis and self._redis_connected:
            try:
                pattern = f"{self.key_prefix}:*"
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
                print(f"[Cache] Cleared {len(keys)} keys from Redis")
            except Exception as e:
                print(f"[Cache] Failed to clear Redis: {e}")
        
        print("[Cache] Memory cache cleared")
    
    def delete(self, cache_key: str):
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
        
        if self._redis_connected:
            try:
                full_key = self._get_full_key(cache_key)
                self._redis_client.delete(full_key)
            except Exception:
                pass
    
    def contains(self, cache_key: str) -> bool:
        with self._lock:
            if cache_key in self._cache:
                return True
        
        if self._redis_connected:
            try:
                full_key = self._get_full_key(cache_key)
                return self._redis_client.exists(full_key) > 0
            except Exception:
                pass
        
        return False
    
    def batch_get(self, cache_keys: List[str]) -> List[Optional[CacheEntry]]:
        return [self.get(key) for key in cache_keys]
    
    def batch_set(self, entries: List[Tuple[str, np.ndarray, Dict[str, Any], Dict[str, Any], int, str]]):
        for cache_key, output_img, metrics, blur_analysis, scale, blur_type in entries:
            self.set(cache_key, output_img, metrics, blur_analysis, scale, blur_type)
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._cache.items())[-count:]
            return [
                {
                    'key': k,
                    'access_count': v.access_count,
                    'created_at': v.created_at,
                    'scale': v.scale,
                    'blur_type': v.blur_type
                }
                for k, v in items
            ]
