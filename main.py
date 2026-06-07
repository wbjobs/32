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
    PerformanceMetrics,
    serialize_metrics,
    serialize_blur_analysis
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
    cache_hit = result.get('cache_hits', [False])[0]
    
    output_bytes = image_to_bytes(output_img, format=output_format.upper())
    
    response = Response(
        content=output_bytes,
        media_type=f"image/{output_format.lower()}"
    )
    
    response.headers["X-Processing-Time-MS"] = str(round(metrics.processing_time_ms, 2))
    response.headers["X-Memory-Usage-MB"] = str(round(metrics.peak_memory_usage_mb, 2))
    response.headers["X-Blur-Type"] = blur_analysis.blur_type.value if hasattr(blur_analysis, 'blur_type') else blur_analysis.get('blur_type', 'unknown')
    response.headers["X-Blur-Confidence"] = str(round(blur_analysis.confidence, 4) if hasattr(blur_analysis, 'confidence') else blur_analysis.get('confidence', 0))
    response.headers["X-Cache-Hit"] = "true" if cache_hit else "false"
    
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
    cache_hit = result.get('cache_hits', [False])[0]
    
    output_bytes = image_to_bytes(output_img, format="PNG")
    output_base64 = base64.b64encode(output_bytes).decode('utf-8')
    
    return {
        "success": True,
        "output_image": f"data:image/png;base64,{output_base64}",
        "metrics": serialize_metrics(metrics),
        "blur_analysis": serialize_blur_analysis(blur_analysis),
        "scale": scale,
        "cache_hit": cache_hit
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
    read_errors = []
    
    for file in files:
        try:
            contents = await file.read()
            img = bytes_to_image(contents)
            images.append(img)
            filenames.append(file.filename)
            read_errors.append(None)
        except Exception as e:
            error_msg = f"Invalid image file {file.filename}: {str(e)}"
            images.append({"error": error_msg})
            filenames.append(file.filename)
            read_errors.append(error_msg)
    
    has_read_errors = any(e is not None for e in read_errors)
    
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
        return JSONResponse(
            status_code=207,
            content={
                "success": False,
                "request_id": request_item.request_id,
                "status": "timeout",
                "message": "Processing timed out. Use /batch/status/{request_id} to check status."
            }
        )
    
    if completed_item.status != RequestStatus.COMPLETED:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "request_id": completed_item.request_id,
                "status": completed_item.status.value,
                "error": completed_item.error
            }
        )
    
    result = completed_item.result
    
    results = []
    has_processing_errors = False
    cache_hits = result.get('cache_hits', [])
    
    for i in range(len(filenames)):
        status = result['statuses'][i]
        error = result['errors'][i]
        cache_hit = cache_hits[i] if i < len(cache_hits) else False
        
        if status == 'success' and result['output_images'][i] is not None:
            output_img = result['output_images'][i]
            output_bytes = image_to_bytes(output_img, format="PNG")
            output_base64 = base64.b64encode(output_bytes).decode('utf-8')
            
            results.append({
                "status": "success",
                "filename": filenames[i],
                "image_data": f"data:image/png;base64,{output_base64}",
                "metrics": serialize_metrics(result['metrics_list'][i]),
                "blur_analysis": serialize_blur_analysis(result['blur_analyses'][i]),
                "cache_hit": cache_hit,
                "error": None
            })
        else:
            has_processing_errors = True
            results.append({
                "status": "failed",
                "filename": filenames[i],
                "image_data": None,
                "metrics": None,
                "blur_analysis": None,
                "cache_hit": False,
                "error": error
            })
    
    has_any_errors = has_read_errors or has_processing_errors
    status_code = 207 if has_any_errors else 200
    
    processing_time_ms = None
    if completed_item.processing_end and completed_item.processing_start:
        processing_time_ms = round(
            (completed_item.processing_end - completed_item.processing_start) * 1000, 2
        )
    
    cache_hit_count = result.get('cache_hit_count', 0)
    
    return JSONResponse(
        status_code=status_code,
        content={
            "success": not has_any_errors,
            "request_id": completed_item.request_id,
            "scale": scale,
            "batch_size": result['batch_size'],
            "success_count": result['success_count'],
            "failed_count": result['failed_count'],
            "cache_hit_count": cache_hit_count,
            "total_processing_time_ms": round(result['total_processing_time_ms'], 2),
            "processing_time_ms": processing_time_ms,
            "results": results
        }
    )


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
            "success_count": item.result.get('success_count', 0),
            "failed_count": item.result.get('failed_count', 0),
            "cache_hit_count": item.result.get('cache_hit_count', 0),
            "total_processing_time_ms": round(item.result.get('total_processing_time_ms', 0), 2)
        }
        
        if 'statuses' in item.result:
            cache_hits = item.result.get('cache_hits', [])
            response["item_statuses"] = [
                {
                    "index": i,
                    "status": s,
                    "cache_hit": cache_hits[i] if i < len(cache_hits) else False,
                    "error": item.result.get('errors', [])[i] if item.result.get('errors') else None
                }
                for i, s in enumerate(item.result['statuses'])
            ]
    
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


@app.get("/cache/stats")
async def get_cache_stats():
    stats = app.state.processor.get_cache_stats()
    return {
        "success": True,
        "cache_enabled": stats is not None,
        "stats": stats if stats else {}
    }


@app.get("/cache/recent")
async def get_cache_recent(count: int = 10):
    recent = app.state.processor.get_cache_recent(count=count)
    return {
        "success": True,
        "count": len(recent),
        "recent": recent
    }


@app.post("/cache/clear")
async def clear_cache(clear_redis: bool = True):
    app.state.processor.clear_cache(clear_redis=clear_redis)
    return {
        "success": True,
        "message": f"Cache cleared (redis: {'yes' if clear_redis else 'no'})"
    }


@app.post("/cache/compute-hash")
async def compute_image_hash(
    file: UploadFile = File(...),
    scale: int = 4
):
    try:
        contents = await file.read()
        img = bytes_to_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    processor = app.state.processor
    if processor.cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    cache_key = processor.cache.compute_key(img, scale)
    
    return {
        "success": True,
        "cache_key": cache_key,
        "scale": scale,
        "in_cache": processor.cache.contains(cache_key)
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
