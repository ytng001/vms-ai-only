#STEP 1 : Run create_html.py if you haven't already to generate the index.html file in the static directory

from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Optional
import queue
import uvicorn
import os
import random
from camera_manager import CameraManager


# Initialize FastAPI app
app = FastAPI(title="CCTV Stream Detection Server")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add static files mounting - now safe since directory exists
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load YOLOv11 model on GPU
model = YOLO("yolo11n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
# Initialize Camera Manager
camera_manager = CameraManager(model)


# Function to read camera streams from file
def load_camera_sources(filename="camera_rtsp.txt"):
    """
    Read camera sources from a text file.
    Expected format: Each line contains <camera_id>:<source_url>
    Example: 
        0:0                # Local camera with index 0
        1:rtsp://user:pass@192.168.1.100:554/stream1
        2:http://camera-url.com/stream
    """
    cameras = []
    try:
        if not os.path.exists(filename):
            print(f"Warning: Camera configuration file '{filename}' not found.")
            return cameras
            
        with open(filename, 'r') as f:
            line_num = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                
                line_num += 1
                    
                try:
                    # Parse "id:source" format
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            camera_id_str, source = parts
                            # Only try to convert to int if it's not an RTSP/HTTP URL
                            if not camera_id_str.startswith(('rtsp', 'http')):
                                try:
                                    camera_id = int(camera_id_str.strip())
                                    source = source.strip()
                                except ValueError:
                                    # If camera_id is not a valid integer, use line number
                                    print(f"Warning: Invalid camera ID: {camera_id_str}, using line number {line_num} as ID")
                                    camera_id = line_num
                                    source = line
                            else:
                                # It's an RTSP URL with a colon but no camera ID
                                camera_id = line_num
                                source = line
                        else:
                            camera_id = line_num
                            source = line
                    else:
                        # If no ID specified, use line number as ID
                        camera_id = line_num
                        source = line
                        
                    # Handle numeric sources
                    if isinstance(source, str) and source.isdigit():
                        source = int(source)
                        
                    cameras.append((camera_id, source))
                            
                except Exception as e:
                    print(f"Warning: Failed to parse camera entry: {line}, error: {e}")
                    
        print(f"Loaded {len(cameras)} cameras from '{filename}'")
        return cameras
        
    except Exception as e:
        print(f"Error loading camera sources: {e}")
        return cameras


# Generate MJPEG frames for streaming
def generate_frames(camera_id: int):
    while True:
        frame = camera_manager.get_frame(camera_id)
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
            
        # Yield the frame in the MJPEG format
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )
        
        # Control frame rate
        time.sleep(0.033)  # ~30 FPS

# API Routes
@app.get("/")
async def root():
    return {"message": "CCTV Stream Detection Server", "version": "1.0.0"}

@app.post("/cameras/start-all")
async def start_all_cameras():
    """Start capturing for all cameras"""
    try:
        # Get list of all camera IDs
        camera_ids = list(camera_manager.streams.keys())
        
        if not camera_ids:
            return JSONResponse(
                status_code=200,
                content={"message": "No cameras available to start", "count": 0}
            )
        
        # Start each camera
        started_count = 0
        for camera_id in camera_ids:
            if camera_manager.start_camera(camera_id):
                started_count += 1
        
        return {
            "message": f"Started {started_count} of {len(camera_ids)} cameras", 
            "count": started_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting cameras: {str(e)}")

@app.post("/cameras/stop-all")
async def stop_all_cameras():
    """Stop capturing for all cameras"""
    try:
        # Get list of all camera IDs
        camera_ids = list(camera_manager.streams.keys())
        
        if not camera_ids:
            return JSONResponse(
                status_code=200,
                content={"message": "No cameras available to stop", "count": 0}
            )
        
        # Stop each camera
        stopped_count = 0
        for camera_id in camera_ids:
            if camera_manager.stop_camera(camera_id):
                stopped_count += 1
        
        return {
            "message": f"Stopped {stopped_count} of {len(camera_ids)} cameras", 
            "count": stopped_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping cameras: {str(e)}")

@app.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: int):
    """Start capturing for a specific camera"""
    success = camera_manager.start_camera(camera_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to start camera {camera_id}")
    return {"message": f"Camera {camera_id} started successfully"}

@app.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: int):
    """Stop capturing for a specific camera"""
    success = camera_manager.stop_camera(camera_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to stop camera {camera_id}")
    return {"message": f"Camera {camera_id} stopped successfully"}

@app.post("/cameras/{camera_id}")
async def add_camera(camera_id: int, source: Optional[str] = None):
    """Add a new camera stream by ID and optional source URL"""
    success = camera_manager.add_camera(camera_id, source)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add camera")
    return {"message": f"Camera {camera_id} added successfully"}

@app.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: int):
    """Remove a camera stream by ID"""
    success = camera_manager.remove_camera(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"message": f"Camera {camera_id} removed successfully"}

@app.get("/cameras")
async def list_cameras():
    """List all active cameras"""
    return {"cameras": camera_manager.get_all_cameras()}

@app.get("/stream/{camera_id}")
async def stream_camera(camera_id: int):
    """Stream a camera feed as MJPEG"""
    if camera_id not in camera_manager.streams:
        # Try to add the camera first if it's numeric
        if isinstance(camera_id, int) and camera_id >= 0:
            success = camera_manager.add_camera(camera_id)
            if not success:
                raise HTTPException(status_code=404, detail="Camera not found or couldn't be initialized")
        else:
            raise HTTPException(status_code=404, detail="Camera not found")
            
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/reload-cameras")
async def reload_cameras():
    """Reload cameras from configuration file"""
    # Remove all existing cameras
    existing_cameras = list(camera_manager.streams.keys())
    for camera_id in existing_cameras:
        camera_manager.remove_camera(camera_id)
    
    # Load cameras from file
    cameras = load_camera_sources()
    added_count = 0
    
    # Add each camera
    for camera_id, source in cameras:
        if camera_manager.add_camera(camera_id, source):
            added_count += 1
    
    return {
        "message": f"Reloaded cameras from configuration file",
        "total": len(cameras),
        "added": added_count
    }

@app.get("/viewer")
async def get_viewer():
    """Serve the main page for the viewer"""
    try:
        # Create static directory if it doesn't exist
        os.makedirs("static", exist_ok=True)
        
        # Check if index.html exists, if not return an error
        if not os.path.exists("static/index.html"):
            return JSONResponse(
                status_code=404,
                content={"message": "index.html not found in static directory"}
            )
        
        # Read the HTML content from the file
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error reading index.html: {str(e)}"}
        )

@app.get("/edit")
async def get_editor():
    """Serve the editor page for selecting camera views"""
    try:
        # Create static directory if it doesn't exist
        os.makedirs("static", exist_ok=True)
        
        # Check if edit.html exists, if not return an error
        if not os.path.exists("static/edit.html"):
            # If edit.html doesn't exist, create it
            with open("static/edit.html", "w", encoding="utf-8") as f:
                f.write(generate_edit_html())
        
        # Read the HTML content from the file
        with open("static/edit.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error reading or creating edit.html: {str(e)}"}
        )

# Startup event to load cameras from file
@app.on_event("startup")
async def startup_event():
    # Load cameras from the configuration file
    cameras = load_camera_sources()
    print ("Starting up server and loading cameras from configuration file...")
    print (cameras)
    # If no cameras in file, add the default camera (0)
    if not cameras:
        print("No cameras found in configuration file. Adding default camera (ID: 0)")
        camera_manager.add_camera(0)
    else:
        # Add all cameras from the configuration file
        for camera_id, source in cameras:
            success = camera_manager.add_camera(camera_id, source)
            if success:
                print(f"Added camera {camera_id} with source {source}")
            else:
                print(f"Failed to add camera {camera_id} with source {source}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)