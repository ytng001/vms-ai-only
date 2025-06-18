import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Optional
import queue
import random
# Camera streams manager
class CameraManager:
    def __init__(self, model, max_cameras=10):
        self.streams: Dict[int, dict] = {}
        self.max_cameras = max_cameras
        self.lock = threading.Lock()
        self.disconnected_cameras = {}  # Store {camera_id: (source, last_attempt_time)}
        self.model = model
        # Start the recovery thread
        self.recovery_active = True
        self.recovery_thread = threading.Thread(
            target=self._recovery_thread,
            daemon=True,
            name=f"CameraRecoveryThread"
        )
        self.recovery_thread.start()
        print("Started camera recovery thread")
        
    def _recovery_thread(self):
        """Background thread to periodically attempt reconnection of disconnected cameras"""
        RECOVERY_INTERVAL = 5  # Seconds between recovery attempts
        
        while self.recovery_active:
            # Sleep at the beginning to allow initial setup
            time.sleep(RECOVERY_INTERVAL)
            
            # Make a copy to avoid modification during iteration
            with self.lock:
                cameras_to_recover = self.disconnected_cameras.copy()
            
            # Try to recover each disconnected camera
            for camera_id, (source, last_attempt) in cameras_to_recover.items():
                current_time = time.time()
                
                # Check if it's time to attempt recovery
                if current_time - last_attempt >= RECOVERY_INTERVAL:
                    print(f"Attempting to recover camera {camera_id} with source {source}")
                    
                    # Attempt to add the camera back
                    success = self.add_camera(camera_id, source)
                    
                    with self.lock:
                        if success:
                            # If successful, remove from disconnected list
                            if camera_id in self.disconnected_cameras:
                                del self.disconnected_cameras[camera_id]
                            print(f"Successfully recovered camera {camera_id}")
                        else:
                            # Update the last attempt time
                            self.disconnected_cameras[camera_id] = (source, current_time)
                            print(f"Failed to recover camera {camera_id}, will retry in {RECOVERY_INTERVAL} seconds")
    
    def add_camera(self, camera_id: int, source: str = None):
        """Add a camera stream by ID, source can be an index or RTSP URL"""
        with self.lock:
            if camera_id in self.streams:
                print(f"Camera {camera_id} already exists")
                return False
                
            # If source is None, use camera_id as index
            if source is None:
                source = camera_id
            
            print(f"Attempting to add camera {camera_id} with source {source}")
                
            # Create camera stream object
            stream = {
                'cap': cv2.VideoCapture(source),
                'frame_queue': queue.Queue(maxsize=10),
                'last_frame': None,
                'active': True,
                'thread': None,
                'fps': 0,
                'last_access': time.time(),
                'source': source
            }
            
            # Check if camera opened successfully
            if not stream['cap'].isOpened():
                print(f"Failed to open camera {camera_id} with source {source}")
                stream['cap'].release()
                
                # Add to disconnected cameras for recovery
                self.disconnected_cameras[camera_id] = (source, time.time())
                return False
                
            # Add to streams dictionary before starting thread
            self.streams[camera_id] = stream
            
            # Start background thread for this camera
            try:
                stream['thread'] = threading.Thread(
                    target=self._camera_thread,
                    args=(camera_id,),
                    daemon=True,
                    name=f"CameraThread-{camera_id}"
                )
                stream['thread'].start()
                print(f"Started thread for camera {camera_id}")
                
                # If this was a recovered camera, remove from disconnected list
                if camera_id in self.disconnected_cameras:
                    del self.disconnected_cameras[camera_id]
                    
                return True
            except Exception as e:
                print(f"Error starting thread for camera {camera_id}: {e}")
                # Clean up on failure
                if camera_id in self.streams:
                    if self.streams[camera_id]['cap'] is not None:
                        self.streams[camera_id]['cap'].release()
                    del self.streams[camera_id]
                
                # Add to disconnected cameras for recovery
                self.disconnected_cameras[camera_id] = (source, time.time())
                return False
    
    def _camera_thread(self, camera_id: int):
        """Background thread for camera processing"""
        stream = self.streams[camera_id]
        cap = stream['cap']
        frame_count = 0
        start_time = time.time()
        
        # Add a frame counter
        # Skip Frames for High FPS Sources
        frame_skip = 20  # Process every nth frame (adjust as needed)
        frame_counter = 0
        
        # Add tracking for processing lag
        processing_lag = 0
        
        # Track consecutive failures
        consecutive_failures = 0
        MAX_FAILURES = 5  # Maximum consecutive failures before marking for recovery
        
        while stream['active']:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"Camera {camera_id}: Failed to read frame ({consecutive_failures}/{MAX_FAILURES})")
                
                # Try immediate reconnection
                retries = 3
                reconnected = False
                while retries > 0 and stream['active']:
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(stream['source'])
                    ret, frame = cap.read()
                    if ret:
                        print(f"Camera {camera_id}: Successfully reconnected")
                        consecutive_failures = 0
                        reconnected = True
                        break
                    retries -= 1
                
                # If we've hit the maximum consecutive failures and couldn't reconnect
                if consecutive_failures >= MAX_FAILURES and not reconnected:
                    print(f"Camera {camera_id}: Too many consecutive failures, marking for recovery")
                    # Add to disconnected cameras list for scheduled recovery
                    with self.lock:
                        stream['active'] = False
                        self.disconnected_cameras[camera_id] = (stream['source'], time.time())
                    break
                
                # Skip the rest of the loop if we couldn't get a frame
                if not reconnected:
                    time.sleep(0.1)  # Small delay to prevent tight loop
                    continue
            else:
                # Reset failure counter on successful frame read
                consecutive_failures = 0
            
            # Calculate lag based on queue size
            processing_lag = stream['frame_queue'].qsize() / 10  # Assuming batch size of 10
            
            # Implement adaptive frame skipping based on lag
            if processing_lag > 2:  # If we're more than 2 batches behind
                skip_probability = min(0.8, processing_lag / 10)  # Cap at 80%
                if random.random() < skip_probability:
                    continue  # Skip this frame
            
            # Only process every nth frame
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue
            
            # Process frame with YOLOv11
            results =  self.model(frame)
            
            # Annotate the frame with detection results
            annotated_frame = self._annotate_frame(frame, results)
            
            # Update the frame queue and last frame
            if stream['frame_queue'].full():
                try:
                    stream['frame_queue'].get_nowait()
                except queue.Empty:
                    pass
            
            try:
                stream['frame_queue'].put_nowait(annotated_frame)
                stream['last_frame'] = annotated_frame
            except queue.Full:
                pass
                
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # Update FPS every second
                stream['fps'] = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                
        # Clean up
        cap.release()
        print(f"Camera {camera_id} thread exiting")
    
    def remove_camera(self, camera_id: int):
        """Remove a camera stream"""
        with self.lock:
            # Remove from active streams if present
            if camera_id in self.streams:
                stream = self.streams[camera_id]
                stream['active'] = False
                if stream['thread'] and stream['thread'].is_alive():
                    stream['thread'].join(timeout=1.0)
                
                stream['cap'].release()
                del self.streams[camera_id]
            
            # Also remove from disconnected cameras if present
            if camera_id in self.disconnected_cameras:
                del self.disconnected_cameras[camera_id]
                
            return True
    
    def start_camera(self, camera_id: int) -> bool:
        """Start capturing for a specific camera if it's stopped"""
        with self.lock:
            if camera_id not in self.streams:
                print(f"Camera {camera_id} doesn't exist")
                return False
            
            stream = self.streams[camera_id]
            if stream['active']:
                # Already active, nothing to do
                print(f"Camera {camera_id} is already active")
                return True
            
            # Reactivate the camera
            print(f"Starting camera {camera_id}")
            stream['active'] = True
            
            # If the thread is dead, create a new one
            if not stream['thread'] or not stream['thread'].is_alive():
                try:
                    # Reopen the capture if needed
                    if not stream['cap'].isOpened():
                        stream['cap'] = cv2.VideoCapture(stream['source'])
                        if not stream['cap'].isOpened():
                            print(f"Failed to reopen camera {camera_id}")
                            return False
                    
                    stream['thread'] = threading.Thread(
                        target=self._camera_thread,
                        args=(camera_id,),
                        daemon=True,
                        name=f"CameraThread-{camera_id}"
                    )
                    stream['thread'].start()
                    print(f"Started new thread for camera {camera_id}")
                except Exception as e:
                    print(f"Error starting new thread for camera {camera_id}: {e}")
                    stream['active'] = False
                    return False
            
            return True

    def stop_camera(self, camera_id: int) -> bool:
        """Stop capturing for a specific camera without removing it"""
        with self.lock:
            if camera_id not in self.streams:
                print(f"Camera {camera_id} doesn't exist")
                return False
            
            stream = self.streams[camera_id]
            if not stream['active']:
                # Already inactive
                print(f"Camera {camera_id} is already stopped")
                return True
            
            # Stop the camera thread by setting active to False
            # The thread will gracefully terminate itself
            print(f"Stopping camera {camera_id}")
            stream['active'] = False
            return True
    
    # Other methods remain the same
    
    def _annotate_frame(self, frame, results):
        """Annotate frame with detection results"""
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
            classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{int(cls)}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
        
    def get_frame(self, camera_id: int):
        """Get the latest processed frame for a camera"""
        if camera_id not in self.streams:
            return None
        
        stream = self.streams[camera_id]
        stream['last_access'] = time.time()
        
        try:
            return stream['frame_queue'].get(timeout=0.5)
        except queue.Empty:
            return stream['last_frame']

    
    def get_all_cameras(self):
        """Get list of all active cameras"""
        with self.lock:
            return [{
                'id': camera_id,
                'source': stream['source'],
                'fps': stream['fps'],
                'last_access': stream['last_access'],
                'active': stream['active']
            } for camera_id, stream in self.streams.items()]