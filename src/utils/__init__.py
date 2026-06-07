from typing import Any, Dict, Optional, Union
from .performance_monitor import PerformanceMonitor, PerformanceMetrics
from .image_utils import (
    bytes_to_image,
    image_to_bytes,
    preprocess_image,
    postprocess_image,
    validate_image_size,
    resize_image,
    calculate_tile_count,
    estimate_memory_usage,
    read_image_file,
    save_image_file,
    generate_test_image
)
from .image_hashing import ImageHasher, compute_image_hash, compute_hash_fast
from src.processing.blur_detection import BlurAnalysis


def serialize_metrics(metrics: PerformanceMetrics) -> Dict[str, Any]:
    return {
        "processing_time_ms": round(metrics.processing_time_ms, 2),
        "peak_memory_usage_mb": round(metrics.peak_memory_usage_mb, 2),
        "gpu_memory_usage_mb": round(metrics.gpu_memory_usage_mb, 2) if metrics.gpu_memory_usage_mb else None,
        "cpu_usage_percent": round(metrics.cpu_usage_percent, 2),
        "input_size": list(metrics.input_size) if metrics.input_size else None,
        "output_size": list(metrics.output_size) if metrics.output_size else None,
        "tile_count": metrics.tile_count,
        "blur_detection_time_ms": round(metrics.blur_detection_time_ms, 2),
        "inference_time_ms": round(metrics.inference_time_ms, 2),
        "preprocessing_time_ms": round(metrics.preprocessing_time_ms, 2),
        "postprocessing_time_ms": round(metrics.postprocessing_time_ms, 2)
    }


def serialize_blur_analysis(analysis: Union[BlurAnalysis, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(analysis, dict):
        return analysis
    
    return {
        "blur_type": analysis.blur_type.value,
        "confidence": round(analysis.confidence, 4),
        "severity": round(analysis.severity, 4),
        "details": {k: round(float(v), 4) for k, v in analysis.details.items()}
    }


__all__ = [
    'PerformanceMonitor',
    'PerformanceMetrics',
    'bytes_to_image',
    'image_to_bytes',
    'preprocess_image',
    'postprocess_image',
    'validate_image_size',
    'resize_image',
    'calculate_tile_count',
    'estimate_memory_usage',
    'read_image_file',
    'save_image_file',
    'generate_test_image',
    'ImageHasher',
    'compute_image_hash',
    'compute_hash_fast',
    'serialize_metrics',
    'serialize_blur_analysis'
]
