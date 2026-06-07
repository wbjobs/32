import time
import psutil
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from contextlib import contextmanager
import numpy as np


@dataclass
class PerformanceMetrics:
    processing_time_ms: float = 0.0
    peak_memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: Optional[float] = None
    cpu_usage_percent: float = 0.0
    input_size: Optional[tuple] = None
    output_size: Optional[tuple] = None
    tile_count: int = 0
    blur_detection_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0


class PerformanceMonitor:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._process = psutil.Process()
        self._metrics_stack: List[PerformanceMetrics] = []
    
    @contextmanager
    def measure(self, metrics: Optional[PerformanceMetrics] = None):
        if metrics is None:
            metrics = PerformanceMetrics()
        
        start_time = time.time()
        start_memory = self._process.memory_info().rss / (1024 * 1024)
        peak_memory = start_memory
        
        gpu_start_memory = None
        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_start_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        
        cpu_start = self._process.cpu_percent()
        
        try:
            self._metrics_stack.append(metrics)
            yield metrics
        finally:
            self._metrics_stack.pop()
            
            end_time = time.time()
            current_memory = self._process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)
            
            cpu_end = self._process.cpu_percent()
            
            metrics.processing_time_ms = (end_time - start_time) * 1000
            metrics.peak_memory_usage_mb = peak_memory - start_memory
            metrics.cpu_usage_percent = cpu_end - cpu_start
            
            if self.device == 'cuda' and torch.cuda.is_available():
                gpu_end_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                metrics.gpu_memory_usage_mb = gpu_end_memory - (gpu_start_memory or 0)
    
    @contextmanager
    def measure_stage(self, stage_name: str):
        if not self._metrics_stack:
            yield
            return
        
        metrics = self._metrics_stack[-1]
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = (time.time() - start_time) * 1000
            
            if stage_name == 'blur_detection':
                metrics.blur_detection_time_ms = elapsed
            elif stage_name == 'inference':
                metrics.inference_time_ms = elapsed
            elif stage_name == 'preprocessing':
                metrics.preprocessing_time_ms = elapsed
            elif stage_name == 'postprocessing':
                metrics.postprocessing_time_ms = elapsed
    
    def get_system_stats(self) -> Dict[str, float]:
        stats = {
            'system_memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
            'system_memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'system_cpu_percent': psutil.cpu_percent(),
            'process_memory_mb': self._process.memory_info().rss / (1024 * 1024)
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
            stats['gpu_memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        
        return stats
    
    def reset_peak_memory(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
