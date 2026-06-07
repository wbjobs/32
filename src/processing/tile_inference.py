import torch
import numpy as np
from typing import Tuple
from config import settings


def split_image_into_tiles(img: np.ndarray, tile_size: int, overlap: int) -> Tuple[np.ndarray, list, Tuple[int, int, int]]:
    h, w, c = img.shape
    tiles = []
    positions = []
    
    stride = tile_size - overlap
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = max(0, y)
            y2 = min(h, y + tile_size)
            x1 = max(0, x)
            x2 = min(w, x + tile_size)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            
            tile = img[y1:y2, x1:x2, :]
            tiles.append(tile)
            positions.append((x1, y1, x2, y2))
    
    return np.array(tiles), positions, (h, w, c)


def merge_tiles_with_blending(tiles: np.ndarray, positions: list, original_shape: Tuple[int, int, int], 
                              scale: int, overlap: int) -> np.ndarray:
    h, w, c = original_shape
    output_h, output_w = h * scale, w * scale
    overlap_scaled = overlap * scale
    stride = (settings.tile_size - overlap) * scale
    
    output = np.zeros((output_h, output_w, c), dtype=np.float32)
    weight_map = np.zeros((output_h, output_w, c), dtype=np.float32)
    
    for i, (pos, tile) in enumerate(zip(positions, tiles)):
        x1, y1, x2, y2 = pos
        x1_s, y1_s = x1 * scale, y1 * scale
        x2_s, y2_s = x2 * scale, y2 * scale
        
        th, tw = tile.shape[:2]
        
        weight = np.ones((th, tw, c), dtype=np.float32)
        
        if x1_s > 0:
            left_overlap = min(overlap_scaled, x1_s - max(0, x1_s - stride))
            left_weights = np.linspace(0, 1, left_overlap).reshape(1, left_overlap, 1)
            weight[:, :left_overlap, :] *= left_weights
        if x2_s < output_w:
            right_overlap = min(overlap_scaled, stride)
            right_weights = np.linspace(1, 0, right_overlap).reshape(1, right_overlap, 1)
            weight[:, -right_overlap:, :] *= right_weights
        if y1_s > 0:
            top_overlap = min(overlap_scaled, y1_s - max(0, y1_s - stride))
            top_weights = np.linspace(0, 1, top_overlap).reshape(top_overlap, 1, 1)
            weight[:top_overlap, :, :] *= top_weights
        if y2_s < output_h:
            bottom_overlap = min(overlap_scaled, stride)
            bottom_weights = np.linspace(1, 0, bottom_overlap).reshape(bottom_overlap, 1, 1)
            weight[-bottom_overlap:, :, :] *= bottom_weights
        
        output[y1_s:y2_s, x1_s:x2_s, :] += tile * weight
        weight_map[y1_s:y2_s, x1_s:x2_s, :] += weight
    
    weight_map[weight_map == 0] = 1
    output = output / weight_map
    
    return np.clip(output, 0, 255).astype(np.uint8)


def tile_inference(model, img_tensor: torch.Tensor, device: str, 
                   tile_size: int = 512, overlap: int = 32, scale: int = 4) -> torch.Tensor:
    batch_size, c, h, w = img_tensor.shape
    stride = tile_size - overlap
    
    output_h, output_w = h * scale, w * scale
    output = torch.zeros((batch_size, c, output_h, output_w), device=device, dtype=torch.float32)
    weight_map = torch.zeros((batch_size, c, output_h, output_w), device=device, dtype=torch.float32)
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = max(0, y)
            y2 = min(h, y + tile_size)
            x1 = max(0, x)
            x2 = min(w, x + tile_size)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            
            tile = img_tensor[:, :, y1:y2, x1:x2]
            
            with torch.no_grad():
                tile_output = model(tile.to(device))
            
            x1_s, y1_s = x1 * scale, y1 * scale
            x2_s, y2_s = x2 * scale, y2 * scale
            th, tw = tile_output.shape[2], tile_output.shape[3]
            
            weight = torch.ones((1, c, th, tw), device=device, dtype=torch.float32)
            overlap_s = overlap * scale
            stride_s = stride * scale
            
            if x1_s > 0:
                left_w = min(overlap_s, x1_s - max(0, x1_s - stride_s))
                left_weights = torch.linspace(0, 1, left_w, device=device).view(1, 1, 1, left_w)
                weight[:, :, :, :left_w] *= left_weights
            if x2_s < output_w:
                right_w = min(overlap_s, stride_s)
                right_weights = torch.linspace(1, 0, right_w, device=device).view(1, 1, 1, right_w)
                weight[:, :, :, -right_w:] *= right_weights
            if y1_s > 0:
                top_w = min(overlap_s, y1_s - max(0, y1_s - stride_s))
                top_weights = torch.linspace(0, 1, top_w, device=device).view(1, 1, top_w, 1)
                weight[:, :, :top_w, :] *= top_weights
            if y2_s < output_h:
                bottom_w = min(overlap_s, stride_s)
                bottom_weights = torch.linspace(1, 0, bottom_w, device=device).view(1, 1, bottom_w, 1)
                weight[:, :, -bottom_w:, :] *= bottom_weights
            
            output[:, :, y1_s:y2_s, x1_s:x2_s] += tile_output * weight
            weight_map[:, :, y1_s:y2_s, x1_s:x2_s] += weight
    
    weight_map[weight_map == 0] = 1
    output = output / weight_map
    
    return output
