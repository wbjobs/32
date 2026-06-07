import numpy as np
import cv2
import torch
from PIL import Image
import io
from typing import Tuple, List, Union
from config import settings


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def image_to_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def preprocess_image(img: np.ndarray, device: str) -> torch.Tensor:
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor.to(device)


def postprocess_image(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def validate_image_size(img: np.ndarray, max_size: int = 2048) -> Tuple[bool, str]:
    h, w = img.shape[:2]
    
    if h > max_size or w > max_size:
        return False, f"Image size {w}x{h} exceeds maximum allowed size {max_size}x{max_size}"
    
    if h < 16 or w < 16:
        return False, f"Image size {w}x{h} is too small. Minimum size is 16x16"
    
    return True, ""


def resize_image(img: np.ndarray, max_size: int = 2048) -> np.ndarray:
    h, w = img.shape[:2]
    
    if h <= max_size and w <= max_size:
        return img
    
    scale = max_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def calculate_tile_count(h: int, w: int, tile_size: int, overlap: int) -> int:
    if h <= tile_size and w <= tile_size:
        return 1
    stride = tile_size - overlap
    tiles_y = max(1, (h - overlap + stride - 1) // stride)
    tiles_x = max(1, (w - overlap + stride - 1) // stride)
    return tiles_y * tiles_x


def estimate_memory_usage(h: int, w: int, tile_size: int, scale: int, device: str) -> float:
    tile_pixels = tile_size * tile_size * 3
    output_pixels = tile_size * scale * tile_size * scale * 3
    model_params = 23 * (64 * 3 * 3 * 3 + 64 * 64 * 3 * 3 * 5 + 64 * 3 * 3 * 3)
    
    bytes_per_pixel = 4
    estimated_memory = (tile_pixels + output_pixels) * bytes_per_pixel + model_params * bytes_per_pixel
    
    return estimated_memory / (1024 * 1024)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def numpy_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def create_padding_mask(h: int, w: int, tile_size: int, overlap: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)
    
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if x > 0:
                mask[:, x:x+overlap] *= np.linspace(0, 1, overlap).reshape(1, -1)
            if x + tile_size < w:
                mask[:, x+tile_size-overlap:x+tile_size] *= np.linspace(1, 0, overlap).reshape(1, -1)
            if y > 0:
                mask[y:y+overlap, :] *= np.linspace(0, 1, overlap).reshape(-1, 1)
            if y + tile_size < h:
                mask[y+tile_size-overlap:y+tile_size, :] *= np.linspace(1, 0, overlap).reshape(-1, 1)
    
    return mask


def read_image_file(filepath: str) -> np.ndarray:
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Cannot read image from {filepath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image_file(img: np.ndarray, filepath: str):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def generate_test_image(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    
    r = np.sin(xx * 4 * np.pi) * 0.5 + 0.5
    g = np.sin(yy * 4 * np.pi) * 0.5 + 0.5
    b = np.sin((xx + yy) * 4 * np.pi) * 0.5 + 0.5
    
    img = np.stack([r, g, b], axis=-1)
    return (img * 255).astype(np.uint8)
