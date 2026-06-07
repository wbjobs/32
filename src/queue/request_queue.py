import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Dict, Any
from enum import Enum
import time
import uuid


class RequestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RequestItem:
    request_id: str
    images: List[Any]
    scale: int
    timestamp: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_start: Optional[float] = None
    processing_end: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestQueue:
    def __init__(self, max_concurrent: int = 10, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        self._queue: Deque[RequestItem] = deque()
        self._processing: Dict[str, RequestItem] = {}
        self._completed: Dict[str, RequestItem] = {}
        
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(lock=self._lock)
        
        self._worker_task: Optional[asyncio.Task] = None
        self._processor = None
        self._is_running = False
    
    def set_processor(self, processor):
        self._processor = processor
    
    async def start(self):
        if self._worker_task is None or self._worker_task.done():
            self._is_running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop(self):
        self._is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def submit(self, images: List[Any], scale: int) -> RequestItem:
        async with self._lock:
            if len(self._queue) + len(self._processing) >= self.max_queue_size:
                raise RuntimeError(f"Queue is full. Max size: {self.max_queue_size}")
            
            request_id = str(uuid.uuid4())
            item = RequestItem(
                request_id=request_id,
                images=images,
                scale=scale
            )
            
            self._queue.append(item)
            self._condition.notify()
            
            return item
    
    async def wait_for_result(self, request_id: str, timeout: Optional[float] = None) -> RequestItem:
        start_time = time.time()
        
        while True:
            async with self._lock:
                if request_id in self._completed:
                    return self._completed[request_id]
                
                found = False
                if request_id in self._processing:
                    found = True
                else:
                    for item in self._queue:
                        if item.request_id == request_id:
                            found = True
                            break
                
                if not found:
                    raise ValueError(f"Request {request_id} not found")
            
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Request {request_id} timed out")
            
            await asyncio.sleep(0.1)
    
    async def get_status(self, request_id: str) -> Optional[RequestItem]:
        async with self._lock:
            if request_id in self._completed:
                return self._completed[request_id]
            if request_id in self._processing:
                return self._processing[request_id]
            for item in self._queue:
                if item.request_id == request_id:
                    return item
        return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "queued": len(self._queue),
                "processing": len(self._processing),
                "completed": len(self._completed),
                "max_concurrent": self.max_concurrent,
                "max_queue_size": self.max_queue_size
            }
    
    async def _worker_loop(self):
        while self._is_running:
            async with self._condition:
                while len(self._queue) == 0 or len(self._processing) >= self.max_concurrent:
                    await self._condition.wait()
            
            await self._process_next()
    
    async def _process_next(self):
        async with self._lock:
            if len(self._queue) == 0 or len(self._processing) >= self.max_concurrent:
                return
            
            item = self._queue.popleft()
            item.status = RequestStatus.PROCESSING
            item.processing_start = time.time()
            self._processing[item.request_id] = item
        
        try:
            result = await self._process_item(item)
            async with self._lock:
                item.result = result
                item.status = RequestStatus.COMPLETED
        except Exception as e:
            async with self._lock:
                item.error = str(e)
                item.status = RequestStatus.FAILED
        finally:
            async with self._lock:
                item.processing_end = time.time()
                
                if item.request_id in self._processing:
                    del self._processing[item.request_id]
                self._completed[item.request_id] = item
                
                if len(self._completed) > 1000:
                    oldest_key = next(iter(self._completed))
                    del self._completed[oldest_key]
                
                self._condition.notify()
    
    async def _process_item(self, item: RequestItem) -> Any:
        if self._processor is None:
            raise RuntimeError("Processor not set")
        
        return await self._processor.process(item.images, item.scale, item.metadata)
