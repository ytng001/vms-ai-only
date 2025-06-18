


import numpy as np
import time
import threading
from typing import Dict, List, Optional
import queue
import uvicorn
import os

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Create index.html if it doesn't exist
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCTV Stream Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            color: #333;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .camera-container {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .camera-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .stream-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .camera-stream {
            max-width: 640px;
            width: 100%;
            border: 1px solid #ccc;
        }
        .controls {
            margin-top: 10px;
        }
        button {
            padding: 5px 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        button.remove {
            background-color: #f44336;
        }
        button.remove:hover {
            background-color: #d32f2f;
        }
    </style>
</head>
<body>
    <h1>CCTV Stream Viewer</h1>
    
    <div class="controls">
        <button id="reload-cameras">Reload Cameras</button>
        <button id="add-camera">Add Camera</button>
    </div>
    
    <div id="cameras-container" class="container">
        <!-- Camera streams will be inserted here -->
    </div>

    <script>
        const baseUrl = window.location.protocol + '//' + window.location.host; // Automatically use correct server URL
        
        // Function to load all available cameras
        async function loadCameras() {
            try {
                const response = await fetch(`${baseUrl}/cameras`);
                const data = await response.json();
                
                const camerasContainer = document.getElementById('cameras-container');
                camerasContainer.innerHTML = ''; // Clear existing cameras
                
                if (data.cameras && data.cameras.length > 0) {
                    data.cameras.forEach(camera => {
                        addCameraToUI(camera.id, camera.fps, camera.source);
                    });
                } else {
                    camerasContainer.innerHTML = '<p>No cameras available</p>';
                }
            } catch (error) {
                console.error('Error loading cameras:', error);
                alert('Error loading cameras. Check the console for details.');
            }
        }
        
        // Function to add a camera to the UI
        function addCameraToUI(cameraId, fps = 0, source = '') {
            const camerasContainer = document.getElementById('cameras-container');
            
            const cameraContainer = document.createElement('div');
            cameraContainer.className = 'camera-container';
            cameraContainer.id = `camera-${cameraId}`;
            
            const header = document.createElement('div');
            header.className = 'camera-header';
            
            const title = document.createElement('h3');
            title.textContent = `Camera ${cameraId}`;
            if (source) {
                title.textContent += ` (Source: ${source})`;
            }
            
            const fpsCounter = document.createElement('span');
            fpsCounter.className = 'fps-counter';
            fpsCounter.textContent = `FPS: ${fps.toFixed(1)}`;
            
            header.appendChild(title);
            header.appendChild(fpsCounter);
            
            const streamContainer = document.createElement('div');
            streamContainer.className = 'stream-container';
            
            // Create img element for MJPEG stream
            const img = document.createElement('img');
            img.className = 'camera-stream';
            img.src = `${baseUrl}/stream/${cameraId}`;
            // Add timestamp to prevent caching
            img.src += `?t=${new Date().getTime()}`;
            
            streamContainer.appendChild(img);
            
            const controls = document.createElement('div');
            controls.className = 'controls';
            
            const removeButton = document.createElement('button');
            removeButton.className = 'remove';
            removeButton.textContent = 'Remove Camera';
            removeButton.onclick = () => removeCamera(cameraId);
            
            controls.appendChild(removeButton);
            
            cameraContainer.appendChild(header);
            cameraContainer.appendChild(streamContainer);
            cameraContainer.appendChild(controls);
            
            camerasContainer.appendChild(cameraContainer);
        }
        
        // Function to reload all cameras
        async function reloadCameras() {
            try {
                const response = await fetch(`${baseUrl}/reload-cameras`);
                const data = await response.json();
                alert(`Reloaded cameras: ${data.added} of ${data.total} cameras added successfully`);
                loadCameras();
            } catch (error) {
                console.error('Error reloading cameras:', error);
                alert('Error reloading cameras. Check the console for details.');
            }
        }
        
        // Function to add a new camera
        async function addCamera() {
            const cameraId = prompt('Enter camera ID (number):');
            if (cameraId === null) return; // User cancelled
            
            const cameraIdNum = parseInt(cameraId);
            if (isNaN(cameraIdNum)) {
                alert('Camera ID must be a number');
                return;
            }
            
            const source = prompt('Enter camera source (URL or number, leave empty to use camera ID):');
            
            try {
                const url = `${baseUrl}/cameras/${cameraIdNum}`;
                const options = {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                };
                
                if (source) {
                    options.body = JSON.stringify({ source });
                }
                
                const response = await fetch(url, options);
                
                if (response.ok) {
                    const data = await response.json();
                    alert(data.message);
                    loadCameras();
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                console.error('Error adding camera:', error);
                alert('Error adding camera. Check the console for details.');
            }
        }
        
        // Function to remove a camera
        async function removeCamera(cameraId) {
            try {
                const response = await fetch(`${baseUrl}/cameras/${cameraId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    const data = await response.json();
                    alert(data.message);
                    
                    // Remove from UI
                    const cameraElement = document.getElementById(`camera-${cameraId}`);
                    if (cameraElement) {
                        cameraElement.remove();
                    }
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                console.error('Error removing camera:', error);
                alert('Error removing camera. Check the console for details.');
            }
        }
        
        // Add event listeners
        document.getElementById('reload-cameras').addEventListener('click', reloadCameras);
        document.getElementById('add-camera').addEventListener('click', addCamera);
        
        // Update FPS counters periodically
        setInterval(async () => {
            try {
                const response = await fetch(`${baseUrl}/cameras`);
                const data = await response.json();
                
                if (data.cameras) {
                    data.cameras.forEach(camera => {
                        const fpsCounter = document.querySelector(`#camera-${camera.id} .fps-counter`);
                        if (fpsCounter) {
                            fpsCounter.textContent = `FPS: ${camera.fps.toFixed(1)}`;
                        }
                    });
                }
            } catch (error) {
                console.error('Error updating FPS:', error);
            }
        }, 2000); // Update every 2 seconds
        
        // Load cameras on page load
        document.addEventListener('DOMContentLoaded', loadCameras);
    </script>
</body>
</html>
"""

with open("static/index.html", "w") as f:
    f.write(html_content)
