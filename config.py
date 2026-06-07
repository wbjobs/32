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

settings = Settings()
