import os
import sys
import base64
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response
import uvicorn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from src.core import SuperResolutionProcessor
from src.queue import RequestQueue, RequestStatus
from src.utils import (
    bytes_to_image,
    image_to_bytes,
    PerformanceMetrics
)
from src.processing import BlurType, BlurAnalysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.processor = SuperResolutionProcessor()
    app.state.request_queue = RequestQueue(
        max_concurrent=settings.max_batch_size,
        max_queue_size=settings.max_queue_size
    )
    app.state.request_queue.set_processor(app.state.processor)
    await app.state.request_queue.start()
    
    if app.state.processor._model_loaded:
        app.state.processor.warmup()
    
    yield
    
    await app.state.request_queue.stop()


app = FastAPI(
    title="Blind Super-Resolution API",
    description="RRDB-based blind super-resolution API service with blur detection",
    version="1.0.0",
    lifespan=lifespan
)


def serialize_metrics(metrics: PerformanceMetrics) -> Dict[str, Any]:
    return {
        "processing_time_ms": round(metrics.processing_time_ms, 2),
        "peak_memory_usage_mb": round(metrics.peak_memory_usage_mb, 2),
        "gpu_memory_usage_mb": round(metrics.gpu_memory_usage_mb, 2) if metrics.gpu_memory_usage_mb else None,
        "cpu_usage_percent": round(metrics.cpu_usage_percent, 2),
        "input_size": metrics.input_size,
        "output_size": metrics.output_size,
        "tile_count": metrics.tile_count,
        "blur_detection_time_ms": round(metrics.blur_detection_time_ms, 2),
        "inference_time_ms": round(metrics.inference_time_ms, 2),
        "preprocessing_time_ms": round(metrics.preprocessing_time_ms, 2),
        "postprocessing_time_ms": round(metrics.postprocessing_time_ms, 2)
    }


def serialize_blur_analysis(analysis: BlurAnalysis) -> Dict[str, Any]:
    return {
        "blur_type": analysis.blur_type.value,
        "confidence": round(analysis.confidence, 4),
        "severity": round(analysis.severity, 4),
        "details": {k: round(float(v), 4) for k, v in analysis.details.items()}
    }


@app.get("/")
async def root():
    return {
        "name": "Blind Super-Resolution API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    processor = app.state.processor
    status = processor.get_system_status()
    queue_stats = await app.state.request_queue.get_queue_stats()
    
    return {
        "status": "healthy",
        "device": status["device"],
        "model_loaded": status["model_loaded"],
        "queue": queue_stats,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status")
async def get_system_status():
    processor = app.state.processor
    queue_stats = await app.state.request_queue.get_queue_stats()
    
    return {
        "system": processor.get_system_status(),
        "queue": queue_stats,
        "settings": {
            "max_image_size": settings.max_image_size,
            "tile_size": settings.tile_size,
            "tile_overlap": settings.tile_overlap,
            "max_batch_size": settings.max_batch_size,
            "max_queue_size": settings.max_queue_size,
            "default_scale": settings.scale_factor
        }
    }


@app.post("/super-resolve")
async def super_resolve(
    file: UploadFile = File(...),
    scale: int = 4,
    output_format: str = "PNG"
):
    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
    
    if output_format.upper() not in ["PNG", "JPEG", "JPG"]:
        raise HTTPException(status_code=400, detail="Output format must be PNG, JPEG, or JPG")
    
    try:
        contents = await file.read()
        img = bytes_to_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    result = await app.state.processor.process([img], scale)
    
    output_img = result['output_images'][0]
    metrics = result['metrics_list'][0]
    blur_analysis = result['blur_analyses'][0]
    
    output_bytes = image_to_bytes(output_img, format=output_format.upper())
    
    response = Response(
        content=output_bytes,
        media_type=f"image/{output_format.lower()}"
    )
    
    response.headers["X-Processing-Time-MS"] = str(round(metrics.processing_time_ms, 2))
    response.headers["X-Memory-Usage-MB"] = str(round(metrics.peak_memory_usage_mb, 2))
    response.headers["X-Blur-Type"] = blur_analysis.blur_type.value
    response.headers["X-Blur-Confidence"] = str(round(blur_analysis.confidence, 4))
    
    return response


@app.post("/super-resolve/json")
async def super_resolve_json(
    file: UploadFile = File(...),
    scale: int = 4
):
    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
    
    try:
        contents = await file.read()
        img = bytes_to_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    result = await app.state.processor.process([img], scale)
    
    output_img = result['output_images'][0]
    metrics = result['metrics_list'][0]
    blur_analysis = result['blur_analyses'][0]
    
    output_bytes = image_to_bytes(output_img, format="PNG")
    output_base64 = base64.b64encode(output_bytes).decode('utf-8')
    
    return {
        "success": True,
        "output_image": f"data:image/png;base64,{output_base64}",
        "metrics": serialize_metrics(metrics),
        "blur_analysis": serialize_blur_analysis(blur_analysis),
        "scale": scale
    }


@app.post("/batch/super-resolve")
async def batch_super_resolve(
    files: List[UploadFile] = File(...),
    scale: int = 4,
    wait: bool = True
):
    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
    
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_batch_size} images allowed per batch"
        )
    
    images = []
    filenames = []
    
    for file in files:
        try:
            contents = await file.read()
            img = bytes_to_image(contents)
            images.append(img)
            filenames.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file {file.filename}: {str(e)}")
    
    request_item = await app.state.request_queue.submit(images, scale)
    
    if not wait:
        return {
            "success": True,
            "request_id": request_item.request_id,
            "status": request_item.status.value,
            "message": "Request queued. Use /batch/status/{request_id} to check status.",
            "queue_position": len(await app.state.request_queue.get_queue_stats())
        }
    
    try:
        completed_item = await app.state.request_queue.wait_for_result(
            request_item.request_id,
            timeout=300.0
        )
    except TimeoutError:
        return {
            "success": False,
            "request_id": request_item.request_id,
            "status": "timeout",
            "message": "Processing timed out. Use /batch/status/{request_id} to check status."
        }
    
    if completed_item.status != RequestStatus.COMPLETED:
        return {
            "success": False,
            "request_id": completed_item.request_id,
            "status": completed_item.status.value,
            "error": completed_item.error
        }
    
    result = completed_item.result
    
    output_images = []
    for i, output_img in enumerate(result['output_images']):
        output_bytes = image_to_bytes(output_img, format="PNG")
        output_base64 = base64.b64encode(output_bytes).decode('utf-8')
        output_images.append({
            "filename": filenames[i],
            "image_data": f"data:image/png;base64,{output_base64}",
            "metrics": serialize_metrics(result['metrics_list'][i]),
            "blur_analysis": serialize_blur_analysis(result['blur_analyses'][i])
        })
    
    return {
        "success": True,
        "request_id": completed_item.request_id,
        "scale": scale,
        "batch_size": result['batch_size'],
        "total_processing_time_ms": round(result['total_processing_time_ms'], 2),
        "processing_time_ms": round((completed_item.processing_end - completed_item.processing_start) * 1000, 2) if completed_item.processing_end and completed_item.processing_start else None,
        "results": output_images
    }


@app.get("/batch/status/{request_id}")
async def get_batch_status(request_id: str):
    item = await app.state.request_queue.get_status(request_id)
    
    if item is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    response = {
        "request_id": item.request_id,
        "status": item.status.value,
        "timestamp": item.timestamp,
        "scale": item.scale,
        "batch_size": len(item.images)
    }
    
    if item.processing_start:
        response["processing_start"] = item.processing_start
    
    if item.processing_end:
        response["processing_end"] = item.processing_end
        response["processing_time_ms"] = round((item.processing_end - item.processing_start) * 1000, 2)
    
    if item.error:
        response["error"] = item.error
    
    if item.status == RequestStatus.COMPLETED and item.result:
        response["summary"] = {
            "batch_size": item.result.get('batch_size', len(item.images)),
            "total_processing_time_ms": round(item.result.get('total_processing_time_ms', 0), 2)
        }
    
    return response


@app.get("/queue/stats")
async def get_queue_stats():
    return await app.state.request_queue.get_queue_stats()


@app.post("/analyze-blur")
async def analyze_image_blur(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = bytes_to_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    from src.processing import analyze_blur
    blur_analysis = analyze_blur(img)
    
    return {
        "success": True,
        "blur_analysis": serialize_blur_analysis(blur_analysis)
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
