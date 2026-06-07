from pydantic import BaseModel, ConfigDict
from typing import Optional

class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_path: str = "models/RRDB_ESRGAN_x4.pth"
    device: str = "cuda"
    max_image_size: int = 2048
    tile_size: int = 512
    tile_overlap: int = 32
    max_batch_size: int = 10
    max_queue_size: int = 100
    scale_factor: int = 4
    gpu_memory_fraction: Optional[float] = 0.9
    
    cache_enabled: bool = True
    cache_max_size: int = 100
    cache_ttl_seconds: int = 86400
    cache_hash_size: int = 8
    cache_key_prefix: str = "sr_cache"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_timeout: int = 5
    redis_enabled: bool = True

settings = Settings()
