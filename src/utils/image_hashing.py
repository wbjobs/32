import numpy as np
import cv2
from typing import Tuple, List
from PIL import Image
import imagehash


class ImageHasher:
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
    
    def dhash(self, img: np.ndarray) -> str:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size), interpolation=cv2.INTER_AREA)
        
        diff = resized[:, 1:] > resized[:, :-1]
        
        hash_bits = []
        for row in diff:
            for bit in row:
                hash_bits.append('1' if bit else '0')
        
        hash_hex = ''
        for i in range(0, len(hash_bits), 4):
            hex_char = hex(int(''.join(hash_bits[i:i+4]), 2))[2:]
            hash_hex += hex_char
        
        return hash_hex
    
    def phash(self, img: np.ndarray) -> str:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        resized = cv2.resize(gray, (self.hash_size * 4, self.hash_size * 4), interpolation=cv2.INTER_AREA)
        resized = np.float32(resized)
        
        dct = cv2.dct(resized)
        dct_low = dct[:self.hash_size, :self.hash_size]
        
        dct_low_flat = dct_low.flatten()
        median = np.median(dct_low_flat)
        
        diff = dct_low_flat > median
        
        hash_bits = ['1' if bit else '0' for bit in diff]
        
        hash_hex = ''
        for i in range(0, len(hash_bits), 4):
            hex_char = hex(int(''.join(hash_bits[i:i+4]), 2))[2:]
            hash_hex += hex_char
        
        return hash_hex
    
    def combined_hash(self, img: np.ndarray) -> str:
        dhash_val = self.dhash(img)
        phash_val = self.phash(img)
        return f"{dhash_val}:{phash_val}"
    
    def perceptual_hash_pil(self, img: np.ndarray) -> str:
        pil_img = Image.fromarray(img)
        hash_val = imagehash.phash(pil_img, hash_size=self.hash_size)
        return str(hash_val)
    
    def average_hash_pil(self, img: np.ndarray) -> str:
        pil_img = Image.fromarray(img)
        hash_val = imagehash.average_hash(pil_img, hash_size=self.hash_size)
        return str(hash_val)
    
    def dhash_pil(self, img: np.ndarray) -> str:
        pil_img = Image.fromarray(img)
        hash_val = imagehash.dhash(pil_img, hash_size=self.hash_size)
        return str(hash_val)
    
    def generate_cache_key(self, img: np.ndarray, scale: int) -> str:
        pil_img = Image.fromarray(img)
        
        phash = imagehash.phash(pil_img, hash_size=self.hash_size)
        dhash = imagehash.dhash(pil_img, hash_size=self.hash_size)
        
        return f"scale_{scale}:phash_{phash}:dhash_{dhash}"
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        if len(hash1) != len(hash2):
            return abs(len(hash1) - len(hash2)) * 4
        
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def is_similar(self, img1: np.ndarray, img2: np.ndarray, threshold: int = 5) -> bool:
        hash1 = self.generate_cache_key(img1, 2)
        hash2 = self.generate_cache_key(img2, 2)
        
        phash1 = hash1.split(':')[1].replace('phash_', '')
        phash2 = hash2.split(':')[1].replace('phash_', '')
        
        dhash1 = hash1.split(':')[2].replace('dhash_', '')
        dhash2 = hash2.split(':')[2].replace('dhash_', '')
        
        dist_p = self.hamming_distance(phash1, phash2)
        dist_d = self.hamming_distance(dhash1, dhash2)
        
        return (dist_p + dist_d) / 2 < threshold


def compute_image_hash(img: np.ndarray, scale: int, hash_size: int = 8) -> str:
    hasher = ImageHasher(hash_size=hash_size)
    return hasher.generate_cache_key(img, scale)


def compute_hash_fast(img: np.ndarray, scale: int) -> str:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    dct_low = dct[:8, :8].flatten()
    median = np.median(dct_low)
    bits = ''.join(['1' if v > median else '0' for v in dct_low])
    
    hash_hex = ''
    for i in range(0, len(bits), 4):
        hash_hex += hex(int(bits[i:i+4], 2))[2:]
    
    return f"scale_{scale}:{hash_hex}"
