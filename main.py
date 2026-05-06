import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of all backend URLs
BACKENDS = [
    "https://pdf-backend-sw75.onrender.com",
    "https://pdf-backend-sc0u.onrender.com",
    "https://pdf-backend-0p97.onrender.com",
    "https://pdf-backend-6qla.onrender.com",
    "https://pdf-backend-32n1.onrender.com",
    "https://pdf-backend-6731.onrender.com",
    "https://pdf-backend-sycn.onrender.com",
    "https://pdf-backend-dvjx.onrender.com",
    "https://pdf-backend-fvqd.onrender.com",
    "https://pdf-backend-1ye1.onrender.com",
    "https://pdf-backend-5iej.onrender.com",
    "https://pdf-backend-0vjn.onrender.com",
]

# Request model matching the main backend
class PromptRequest(BaseModel):
    prompt: str


class BackendManager:
    def __init__(self, backends: list[str]):
        self.backends = backends
        self.busy = {backend: False for backend in backends}
        self.lock = asyncio.Lock()
        self.last_used = {backend: 0.0 for backend in backends}
        self.request_count = {backend: 0 for backend in backends}
    
    async def get_available_backend(self, timeout: float = 120.0) -> Optional[str]:
        """Get an available backend, waiting if all are busy."""
        start_time = time.time()
        
        while True:
            async with self.lock:
                # Find available backends sorted by last used time (LRU)
                available = [b for b in self.backends if not self.busy[b]]
                
                if available:
                    # Sort by last used time to distribute load
                    available.sort(key=lambda b: self.last_used[b])
                    backend = available[0]
                    self.busy[backend] = True
                    self.last_used[backend] = time.time()
                    self.request_count[backend] += 1
                    logger.info(f"Assigned backend: {backend}")
                    return backend
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Timeout waiting for available backend after {elapsed:.1f}s")
                return None
            
            # Log waiting status periodically
            if int(elapsed) % 10 == 0 and elapsed > 0:
                busy_count = sum(1 for b in self.backends if self.busy[b])
                logger.info(f"Waiting for backend... {busy_count}/{len(self.backends)} busy, waited {elapsed:.1f}s")
            
            # Wait before checking again
            await asyncio.sleep(0.5)
    
    async def release_backend(self, backend: str):
        """Mark a backend as available again."""
        async with self.lock:
            self.busy[backend] = False
            logger.info(f"Released backend: {backend}")
    
    def get_status(self) -> dict:
        """Get current status of all backends."""
        return {
            "total": len(self.backends),
            "available": sum(1 for b in self.backends if not self.busy[b]),
            "busy": sum(1 for b in self.backends if self.busy[b]),
            "backends": [
                {
                    "url": b,
                    "busy": self.busy[b],
                    "last_used": self.last_used[b],
                    "request_count": self.request_count[b]
                }
                for b in self.backends
            ]
        }


# Initialize backend manager
backend_manager = BackendManager(BACKENDS)

# Create FastAPI app
app = FastAPI(
    title="DeepSeek Middleware",
    description="Load balancer for DeepSeek scraper backends"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    status = backend_manager.get_status()
    return {
        "status": "healthy",
        "service": "deepseek-middleware",
        "backends_available": status["available"],
        "backends_total": status["total"],
        "timestamp": int(time.time())
    }


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    status = backend_manager.get_status()
    return {
        "status": "healthy" if status["available"] > 0 else "degraded",
        "service": "deepseek-middleware",
        "available_backends": status["available"],
        "total_backends": status["total"],
        "timestamp": int(time.time())
    }


@app.get("/status")
async def status():
    """Get detailed status of all backends."""
    return backend_manager.get_status()


@app.post("/scrape-deepseek")
async def scrape_deepseek(req: PromptRequest):
    """
    Main endpoint that proxies requests to available backends.
    Matches the main backend's /scrape-deepseek endpoint.
    """
    # Get an available backend
    backend = await backend_manager.get_available_backend(timeout=120.0)
    
    if not backend:
        raise HTTPException(
            status_code=503,
            detail="All backends are busy. Please try again later."
        )
    
    try:
        logger.info(f"Forwarding request to {backend}")
        
        # Forward the request to the selected backend
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            response = await client.post(
                f"{backend}/scrape-deepseek",
                json={"prompt": req.prompt},
                headers={"Content-Type": "application/json"}
            )
        
        # Return the response from the backend
        if response.status_code == 200:
            return response.json()
        else:
            # Forward error responses
            raise HTTPException(
                status_code=response.status_code,
                detail=response.json().get("detail", "Backend error")
            )
    
    except httpx.TimeoutException:
        logger.error(f"Timeout from backend {backend}")
        raise HTTPException(
            status_code=504,
            detail="Backend request timed out"
        )
    except httpx.RequestError as e:
        logger.error(f"Request error to backend {backend}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to backend: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error from backend {backend}: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        # Always release the backend when done
        await backend_manager.release_backend(backend)


@app.post("/wake-all")
async def wake_all_backends():
    """
    Wake up all backends by hitting their health endpoints.
    Useful for Render free tier which sleeps after inactivity.
    """
    results = []
    
    async def check_health(backend: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.get(f"{backend}/health")
                return {
                    "backend": backend,
                    "status": "awake" if response.status_code == 200 else "error",
                    "response": response.json() if response.status_code == 200 else None
                }
        except Exception as e:
            return {
                "backend": backend,
                "status": "error",
                "error": str(e)
            }
    
    # Wake all backends in parallel
    tasks = [check_health(backend) for backend in BACKENDS]
    results = await asyncio.gather(*tasks)
    
    awake_count = sum(1 for r in results if r["status"] == "awake")
    
    return {
        "message": f"Woke {awake_count}/{len(BACKENDS)} backends",
        "results": results
    }


@app.get("/queue-position")
async def queue_position():
    """Get current queue information."""
    status = backend_manager.get_status()
    return {
        "available_backends": status["available"],
        "busy_backends": status["busy"],
        "total_backends": status["total"],
        "estimated_wait": "immediate" if status["available"] > 0 else "up to 10 minutes"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
