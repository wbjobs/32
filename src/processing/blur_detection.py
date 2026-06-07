import numpy as np
import cv2
from typing import Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum


class BlurType(str, Enum):
    GAUSSIAN = "gaussian"
    MOTION = "motion"
    DOWNSAMPLE = "downsample"
    SHARP = "sharp"
    UNKNOWN = "unknown"


@dataclass
class BlurAnalysis:
    blur_type: BlurType
    confidence: float
    severity: float
    details: Dict[str, float]


def compute_laplacian_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(laplacian)


def compute_fft_analysis(img: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    center_radius = min(h, w) // 10
    outer_radius = min(h, w) // 3
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    center_mask = dist_from_center <= center_radius
    outer_mask = (dist_from_center > center_radius) & (dist_from_center <= outer_radius)
    
    center_energy = np.mean(magnitude[center_mask])
    outer_energy = np.mean(magnitude[outer_mask])
    
    energy_ratio = center_energy / (outer_energy + 1e-10)
    
    angles = np.linspace(0, np.pi, 36)
    directional_energies = []
    
    for angle in angles:
        line_length = min(h, w) // 2
        x1 = int(cx - line_length * np.cos(angle))
        y1 = int(cy - line_length * np.sin(angle))
        x2 = int(cx + line_length * np.cos(angle))
        y2 = int(cy + line_length * np.sin(angle))
        
        mask = np.zeros(magnitude.shape, dtype=np.uint8)
        mask = np.ascontiguousarray(mask)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
        directional_energies.append(np.mean(magnitude[mask > 0]))
    
    directional_std = np.std(directional_energies)
    directional_max = np.max(directional_energies)
    directional_min = np.min(directional_energies)
    directionality = (directional_max - directional_min) / (directional_mean + 1e-10) if (directional_mean := np.mean(directional_energies)) > 0 else 0
    
    return {
        "energy_ratio": energy_ratio,
        "directional_std": directional_std,
        "directionality": directionality,
        "center_energy": center_energy,
        "outer_energy": outer_energy
    }


def compute_edge_density(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])


def compute_dct_analysis(img: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    gray = cv2.resize(gray, (64, 64)).astype(np.float32)
    
    dct = cv2.dct(gray)
    h, w = dct.shape
    
    diagonal_energies = []
    for k in range(1, h + w - 1):
        diag_sum = 0
        count = 0
        for i in range(max(0, k - w + 1), min(h, k + 1)):
            j = k - i
            if 0 <= j < w:
                diag_sum += abs(dct[i, j])
                count += 1
        if count > 0:
            diagonal_energies.append(diag_sum / count)
    
    if len(diagonal_energies) > 10:
        high_freq_energy = np.mean(diagonal_energies[-10:])
        low_freq_energy = np.mean(diagonal_energies[:10])
        energy_decay = np.exp(-np.mean(np.gradient(np.log(np.array(diagonal_energies) + 1e-10))))
    else:
        high_freq_energy = 0
        low_freq_energy = 1
        energy_decay = 0
    
    zigzag_energy = []
    for i in range(h):
        for j in range(w):
            if i + j < h:
                zigzag_energy.append(abs(dct[i, j]))
    
    if len(zigzag_energy) > 0:
        mid = len(zigzag_energy) // 2
        high_freq = np.mean(zigzag_energy[mid:]) if mid < len(zigzag_energy) else 0
        low_freq = np.mean(zigzag_energy[:mid]) if mid > 0 else 1
        hf_ratio = high_freq / (low_freq + 1e-10)
    else:
        hf_ratio = 0
    
    return {
        "high_low_ratio": hf_ratio,
        "energy_decay": energy_decay,
        "high_freq_energy": high_freq_energy
    }


def detect_aliasing(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = min(h, w) / 2
    
    ring_width = max_dist / 20
    ring_energies = []
    
    for i in range(20):
        inner_r = i * ring_width
        outer_r = (i + 1) * ring_width
        mask = (dist_from_center >= inner_r) & (dist_from_center < outer_r)
        if np.any(mask):
            ring_energies.append(np.mean(magnitude[mask]))
        else:
            ring_energies.append(0)
    
    if len(ring_energies) > 5:
        high_freq_rings = ring_energies[10:]
        energy_spikes = np.std(high_freq_rings) / (np.mean(high_freq_rings) + 1e-10)
    else:
        energy_spikes = 0
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_dir = np.arctan2(sobel_y, sobel_x)
    
    hist, _ = np.histogram(grad_dir[grad_mag > np.mean(grad_mag)], bins=36)
    direction_sparsity = np.std(hist) / (np.mean(hist) + 1e-10)
    
    return (energy_spikes + direction_sparsity) / 2


def analyze_blur(img: np.ndarray) -> BlurAnalysis:
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    laplacian_var = compute_laplacian_variance(img)
    fft_result = compute_fft_analysis(img)
    edge_density = compute_edge_density(img)
    dct_result = compute_dct_analysis(img)
    aliasing_score = detect_aliasing(img)
    
    details = {
        "laplacian_variance": laplacian_var,
        "fft_energy_ratio": fft_result["energy_ratio"],
        "fft_directionality": fft_result["directionality"],
        "edge_density": edge_density,
        "dct_high_low_ratio": dct_result["high_low_ratio"],
        "dct_energy_decay": dct_result["energy_decay"],
        "aliasing_score": aliasing_score
    }
    
    is_blurry = laplacian_var < 100 or edge_density < 0.05
    
    if not is_blurry:
        return BlurAnalysis(
            blur_type=BlurType.SHARP,
            confidence=0.9,
            severity=0.1,
            details=details
        )
    
    severity = max(0, min(1, 1 - laplacian_var / 500))
    
    motion_score = fft_result["directionality"] * 0.6 + (1 - fft_result["energy_ratio"] / 5) * 0.4
    gaussian_score = (1 - dct_result["high_low_ratio"]) * 0.5 + (1 - edge_density / 0.1) * 0.3 + (fft_result["energy_ratio"] / 10) * 0.2
    downsample_score = aliasing_score * 0.5 + dct_result["energy_decay"] * 0.3 + (fft_result["energy_ratio"] / 8) * 0.2
    
    scores = {
        BlurType.GAUSSIAN: gaussian_score,
        BlurType.MOTION: motion_score,
        BlurType.DOWNSAMPLE: downsample_score
    }
    
    blur_type = max(scores, key=scores.get)
    confidence = scores[blur_type] / (sum(scores.values()) + 1e-10)
    
    return BlurAnalysis(
        blur_type=blur_type,
        confidence=float(confidence),
        severity=float(severity),
        details=details
    )


def analyze_batch(images: List[np.ndarray]) -> List[BlurAnalysis]:
    return [analyze_blur(img) for img in images]
