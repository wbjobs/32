import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import os

from config import settings
from src.models import load_rrdb_model
from src.processing import tile_inference, analyze_blur, BlurAnalysis
from src.utils import (
    PerformanceMonitor,
    PerformanceMetrics,
    preprocess_image,
    postprocess_image,
    validate_image_size,
    resize_image,
    calculate_tile_count
)


class SuperResolutionProcessor:
    def __init__(self, model_path: str = None, device: str = None, scale: int = None):
        self.model_path = model_path or settings.model_path
        self.device = device or settings.device
        self.scale = scale or settings.scale_factor
        
        self.tile_size = settings.tile_size
        self.tile_overlap = settings.tile_overlap
        self.max_image_size = settings.max_image_size
        
        self.model = None
        self.performance_monitor = None
        self._model_loaded = False
        
        self._init_model()
        self._init_performance_monitor()
    
    def _init_model(self):
        if os.path.exists(self.model_path):
            self.model, self.device = load_rrdb_model(
                self.model_path,
                scale=self.scale,
                device=self.device
            )
            self._model_loaded = True
        else:
            print(f"Warning: Model file not found at {self.model_path}")
            print("Using CPU fallback mode for testing...")
            self._model_loaded = False
            self.device = 'cpu'
    
    def _init_performance_monitor(self):
        self.performance_monitor = PerformanceMonitor(device=self.device)
    
    def _fallback_upscale(self, img: np.ndarray, scale: int) -> np.ndarray:
        h, w = img.shape[:2]
        new_h, new_w = h * scale, w * scale
        return torch.nn.functional.interpolate(
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).numpy() * 255
    
    async def process(self, images: List[np.ndarray], scale: int, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if metadata is None:
            metadata = {}
        
        results = []
        all_metrics = []
        all_blur_analyses = []
        errors = []
        statuses = []
        
        for i, img in enumerate(images):
            if isinstance(img, dict) and img.get('error'):
                results.append(None)
                all_metrics.append(None)
                all_blur_analyses.append(None)
                errors.append(img['error'])
                statuses.append('failed')
                continue
            
            try:
                result = await self._process_single_image(img, scale)
                results.append(result['output_image'])
                all_metrics.append(result['metrics'])
                all_blur_analyses.append(result['blur_analysis'])
                errors.append(None)
                statuses.append('success')
            except Exception as e:
                results.append(None)
                all_metrics.append(None)
                all_blur_analyses.append(None)
                errors.append(str(e))
                statuses.append('failed')
        
        success_count = sum(1 for s in statuses if s == 'success')
        failed_count = sum(1 for s in statuses if s == 'failed')
        total_metrics_time = sum(
            m.processing_time_ms for m in all_metrics if m is not None
        )
        
        return {
            'output_images': results,
            'metrics_list': all_metrics,
            'blur_analyses': all_blur_analyses,
            'errors': errors,
            'statuses': statuses,
            'success_count': success_count,
            'failed_count': failed_count,
            'batch_size': len(images),
            'total_processing_time_ms': total_metrics_time
        }
    
    async def _process_single_image(self, img: np.ndarray, scale: int) -> Dict[str, Any]:
        metrics = PerformanceMetrics()
        
        with self.performance_monitor.measure(metrics):
            with self.performance_monitor.measure_stage('preprocessing'):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                
                h, w = img.shape[:2]
                metrics.input_size = (h, w)
                
                valid, msg = validate_image_size(img, self.max_image_size)
                if not valid:
                    if "exceeds maximum" in msg:
                        img = resize_image(img, self.max_image_size)
                        h, w = img.shape[:2]
                    else:
                        raise ValueError(msg)
                
                tile_count = calculate_tile_count(h, w, self.tile_size, self.tile_overlap)
                metrics.tile_count = tile_count
            
            with self.performance_monitor.measure_stage('blur_detection'):
                blur_analysis = analyze_blur(img)
            
            with self.performance_monitor.measure_stage('inference'):
                if self._model_loaded:
                    img_tensor = preprocess_image(img, self.device)
                    
                    output_tensor = tile_inference(
                        self.model,
                        img_tensor,
                        self.device,
                        tile_size=self.tile_size,
                        overlap=self.tile_overlap,
                        scale=self.scale if scale == 0 else scale
                    )
                    
                    output_img = postprocess_image(output_tensor)
                else:
                    actual_scale = self.scale if scale == 0 else scale
                    output_img = self._fallback_upscale(img, actual_scale)
            
            with self.performance_monitor.measure_stage('postprocessing'):
                output_h, output_w = output_img.shape[:2]
                metrics.output_size = (output_h, output_w)
        
        return {
            'output_image': output_img,
            'metrics': metrics,
            'blur_analysis': blur_analysis
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            'device': self.device,
            'model_loaded': self._model_loaded,
            'model_path': self.model_path,
            'scale': self.scale,
            'tile_size': self.tile_size,
            'tile_overlap': self.tile_overlap,
            'max_image_size': self.max_image_size,
            'system_stats': self.performance_monitor.get_system_stats() if self.performance_monitor else {}
        }
    
    def warmup(self, test_size: Tuple[int, int] = (256, 256)):
        if not self._model_loaded:
            print("Model not loaded, skipping warmup")
            return
        
        print(f"Warming up model with test image {test_size[0]}x{test_size[1]}...")
        
        test_img = np.random.randint(0, 255, (*test_size, 3), dtype=np.uint8)
        test_tensor = preprocess_image(test_img, self.device)
        
        with torch.no_grad():
            for _ in range(3):
                _ = tile_inference(
                    self.model,
                    test_tensor,
                    self.device,
                    tile_size=self.tile_size,
                    overlap=self.tile_overlap,
                    scale=self.scale
                )
        
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        print("Warmup complete")
