import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Optional
import queue
import random
import json
import os

# Camera streams manager
class CameraManager:
    def __init__(self, model, max_cameras=10, masks_config="camera_masks.json"):
        self.streams: Dict[int, dict] = {}
        self.max_cameras = max_cameras
        self.lock = threading.Lock()
        self.disconnected_cameras = {}  # Store {camera_id: (source, last_attempt_time)}
        self.model = model
        self.masks = {}  # Store masks for each camera {camera_id: mask_array}
        self.masks_config = masks_config  # Path to masks configuration file
        self._mask_configs = {}  # Store loaded mask configurations
        
        # Load any saved masks
        self._load_masks()
        
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
                            
                            # Apply saved mask if available
                            self._apply_mask_when_ready(camera_id)
                        else:
                            # Update the last attempt time
                            self.disconnected_cameras[camera_id] = (source, current_time)
                            print(f"Failed to recover camera {camera_id}, will retry in {RECOVERY_INTERVAL} seconds")
    
    def _load_masks(self):
        """Load mask configurations from JSON file"""
        try:
            if os.path.exists(self.masks_config):
                with open(self.masks_config, 'r') as f:
                    mask_data = json.load(f)
                
                print(f"Loading masks from {self.masks_config}")
                # Store mask data temporarily until cameras are initialized
                self._mask_configs = mask_data
                print(f"Loaded {len(mask_data)} mask configurations")
            else:
                print(f"No mask configuration file found at {self.masks_config}")
                self._mask_configs = {}
        except Exception as e:
            print(f"Error loading mask configurations: {e}")
            self._mask_configs = {}
    
    def _save_masks(self):
        """Save mask configurations to JSON file"""
        try:
            mask_data = {}
            
            # For each camera with a mask, store the mask points and dimensions
            with self.lock:
                for camera_id, mask in self.masks.items():
                    # Extract contours from the mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Only save if there are contours
                    if contours:
                        # Get the largest contour (should be the mask)
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Convert contour to list of points
                        points = largest_contour.reshape(-1, 2).tolist()
                        
                        # Get frame dimensions
                        if camera_id in self.streams and self.streams[camera_id].get('last_frame') is not None:
                            height, width = self.streams[camera_id]['last_frame'].shape[:2]
                        else:
                            height, width = mask.shape[:2]
                        
                        # Store mask data
                        mask_data[str(camera_id)] = {
                            'points': points,
                            'frame_width': width,
                            'frame_height': height
                        }
            
            # Save to file
            with open(self.masks_config, 'w') as f:
                json.dump(mask_data, f, indent=4)
            
            # Also update the in-memory mask configs
            self._mask_configs = mask_data
            
            print(f"Saved {len(mask_data)} mask configurations to {self.masks_config}")
        except Exception as e:
            print(f"Error saving mask configurations: {e}")
    
    def _apply_mask_when_ready(self, camera_id: int):
        """Schedule mask application to try repeatedly until successful"""
        def delayed_apply(camera_id):
            # Try up to 10 times with a delay between attempts
            for _ in range(10):
                if self.apply_saved_mask(camera_id):
                    print(f"Successfully applied saved mask to camera {camera_id}")
                    return
                # Wait and try again
                time.sleep(1)
            print(f"Failed to apply mask to camera {camera_id} after multiple attempts")
            
        # Start a thread to handle delayed application of the mask
        thread = threading.Thread(
            target=delayed_apply, 
            args=(camera_id,),
            daemon=True
        )
        thread.start()
    
    def apply_saved_mask(self, camera_id: int):
        """Apply a previously saved mask to a camera if available"""
        camera_id_str = str(camera_id)
        if camera_id_str not in self._mask_configs:
            return False
            
        try:
            with self.lock:
                if camera_id not in self.streams:
                    print(f"Cannot apply saved mask: Camera {camera_id} doesn't exist")
                    return False
                
                stream = self.streams[camera_id]
                if stream.get('last_frame') is None:
                    print(f"Cannot apply saved mask: Camera {camera_id} has no frames yet")
                    return False
                
                # Get actual frame dimensions
                frame_height, frame_width = stream['last_frame'].shape[:2]
                
                # Get saved mask data
                mask_config = self._mask_configs[camera_id_str]
                saved_points = mask_config['points']
                saved_width = mask_config['frame_width']
                saved_height = mask_config['frame_height']
                
                # Scale points if frame dimensions have changed
                scaled_points = []
                scale_x = frame_width / saved_width
                scale_y = frame_height / saved_height
                
                for x, y in saved_points:
                    scaled_x = int(x * scale_x)
                    scaled_y = int(y * scale_y)
                    scaled_points.append((scaled_x, scaled_y))
                
                # Create mask from scaled points
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                points_array = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points_array], 255)
                
                # Save the mask
                self.masks[camera_id] = mask
                print(f"Applied saved mask to camera {camera_id}")
                
                # Force refresh the current frame with mask
                if stream.get('last_frame') is not None:
                    try:
                        current_frame = stream['last_frame'].copy()
                        # Apply mask to current frame
                        masked_frame = self._annotate_frame(current_frame, [], camera_id)
                        
                        # Update last frame and queue
                        stream['last_frame'] = masked_frame
                        
                        # Clear and update queue
                        while not stream['frame_queue'].empty():
                            try:
                                stream['frame_queue'].get_nowait()
                            except queue.Empty:
                                break
                        
                        # Add the masked frame to the queue
                        try:
                            stream['frame_queue'].put_nowait(masked_frame)
                        except queue.Full:
                            pass
                    except Exception as e:
                        print(f"Error applying mask to current frame: {e}")
                
                return True
        except Exception as e:
            print(f"Error applying saved mask to camera {camera_id}: {e}")
                
        return False
    
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
                
                # Apply saved mask if available with retry logic
                if str(camera_id) in self._mask_configs:
                    self._apply_mask_when_ready(camera_id)
                    
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
        frame_skip = 5  # Process every nth frame (adjust as needed)
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
            
            # First apply the mask if it exists
            processed_frame = frame.copy()
            masked_frame = None
            
            # Check if there's a mask for this camera
            if camera_id in self.masks:
                # Get the mask and create a masked version of the frame
                mask = self.masks[camera_id]
                
                # Apply the mask to the frame for detection
                if mask is not None:
                    # Create a copy for mask application
                    masked_frame = frame.copy()
                    
                    # Apply mask to frame - only keep the masked area
                    masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mask)
                    
                    # Use the masked frame for detection
                    processed_frame = masked_frame
                    
            # Process frame with YOLOv11 using the masked frame if available
            results = self.model(processed_frame)
            
            # Annotate the original frame with detection results
            annotated_frame = self._annotate_frame(frame, results, camera_id)
            
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
            
            # Also remove any associated mask
            if camera_id in self.masks:
                del self.masks[camera_id]
                # Update the masks file to reflect the removal
                self._save_masks()
                
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
                    
                    # Apply saved mask if available with retry logic
                    if str(camera_id) in self._mask_configs:
                        self._apply_mask_when_ready(camera_id)
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
    
    def set_mask(self, camera_id: int, mask_points: list, canvas_width: int = None, canvas_height: int = None):
        """Set a mask for a camera using polygon points"""
        with self.lock:
            if camera_id not in self.streams:
                print(f"Cannot set mask: Camera {camera_id} doesn't exist")
                return False
            
            stream = self.streams[camera_id]
            if stream.get('last_frame') is None:
                print(f"Cannot set mask: Camera {camera_id} has no frames yet")
                return False
            
            # Get actual frame dimensions
            frame_height, frame_width = stream['last_frame'].shape[:2]
            print(f"Frame dimensions: {frame_width}x{frame_height}")
            
            # Scale points if canvas dimensions were provided
            scaled_points = []
            if canvas_width and canvas_height:
                print(f"Canvas dimensions: {canvas_width}x{canvas_height}")
                # Calculate scaling factors
                scale_x = frame_width / canvas_width
                scale_y = frame_height / canvas_height
                print(f"Scaling factors: x={scale_x}, y={scale_y}")
                
                # Scale each point
                for x, y in mask_points:
                    scaled_x = int(x * scale_x)
                    scaled_y = int(y * scale_y)
                    scaled_points.append((scaled_x, scaled_y))
                print(f"Scaled {len(mask_points)} points")
            else:
                # Use points as-is if no canvas dimensions provided
                scaled_points = mask_points
            
            try:
                # Create a mask from scaled points
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                
                # Convert points to numpy array of shape required by fillPoly
                points_array = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points_array], 255)
                
                # Save the mask
                self.masks[camera_id] = mask
                print(f"Mask set for camera {camera_id} with {len(scaled_points)} points")
                
                # Also store the original points and canvas dimensions for future reference
                mask_config = {
                    'points': mask_points,
                    'frame_width': frame_width,
                    'frame_height': frame_height
                }
                if canvas_width and canvas_height:
                    mask_config['canvas_width'] = canvas_width
                    mask_config['canvas_height'] = canvas_height
                
                self._mask_configs[str(camera_id)] = mask_config
                
                # Force refresh the current frame with mask
                if stream.get('last_frame') is not None:
                    try:
                        current_frame = stream['last_frame'].copy()
                        # Apply mask to current frame
                        masked_frame = self._annotate_frame(current_frame, [], camera_id)
                        
                        # Update last frame and queue
                        stream['last_frame'] = masked_frame
                        
                        # Clear and update queue
                        while not stream['frame_queue'].empty():
                            try:
                                stream['frame_queue'].get_nowait()
                            except queue.Empty:
                                break
                        
                        # Add the masked frame to the queue
                        try:
                            stream['frame_queue'].put_nowait(masked_frame)
                        except queue.Full:
                            pass
                    except Exception as e:
                        print(f"Error applying mask to current frame: {e}")
                
                return True
            except Exception as e:
                print(f"Error creating mask: {e}")
                return False

    def clear_mask(self, camera_id: int):
        """Clear the mask for a camera"""
        with self.lock:
            # Remove mask from memory
            if camera_id in self.masks:
                del self.masks[camera_id]
                print(f"Mask cleared for camera {camera_id}")
                
                # Also remove from mask configs
                if str(camera_id) in self._mask_configs:
                    del self._mask_configs[str(camera_id)]
                
                # Force refresh the current frame without mask
                if camera_id in self.streams:
                    stream = self.streams[camera_id]
                    if stream.get('last_frame') is not None:
                        try:
                            current_frame = stream['cap'].read()[1]  # Get a new frame
                            if current_frame is not None:
                                # Process with model
                                results = self.model(current_frame)
                                # Annotate without mask
                                annotated_frame = self._annotate_frame(current_frame, results, camera_id)
                                
                                # Update last frame and queue
                                stream['last_frame'] = annotated_frame
                                
                                # Clear and update queue
                                while not stream['frame_queue'].empty():
                                    try:
                                        stream['frame_queue'].get_nowait()
                                    except queue.Empty:
                                        break
                                
                                # Add the new frame to the queue
                                try:
                                    stream['frame_queue'].put_nowait(annotated_frame)
                                except queue.Full:
                                    pass
                        except Exception as e:
                            print(f"Error refreshing frame after clearing mask: {e}")
                
                return True
            return False
    
    def _annotate_frame(self, frame, results, camera_id=None):
        """Annotate frame with detection results and apply mask if present"""
        # If camera_id wasn't provided, try to determine it
        if camera_id is None:
            for cid, stream in self.streams.items():
                if stream.get('last_frame') is frame or id(stream.get('last_frame')) == id(frame):
                    camera_id = cid
                    break
        
        # Make sure we have a copy of the frame to avoid modifying the original
        frame = frame.copy()
        
        # Apply the mask if it exists for this camera
        if camera_id is not None and camera_id in self.masks:
            # Create a copy for the mask application
            original_frame = frame.copy()
            
            # Create a fully black version for areas outside the mask
            black_frame = np.zeros_like(original_frame)
            
            # Get the mask and its inverse
            mask = self.masks[camera_id]
            inv_mask = cv2.bitwise_not(mask)
            
            # Apply the masks - keep only the areas inside the mask, black out the rest
            masked_area = cv2.bitwise_and(original_frame, original_frame, mask=mask)
            black_area = cv2.bitwise_and(black_frame, black_frame, mask=inv_mask)
            
            # Combine: masked area + black area
            frame = cv2.add(masked_area, black_area)
            
            # Draw the mask boundary for better visibility
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            # Add text to indicate a mask is active
            cv2.putText(frame, "MASK ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Process detection results
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
                'active': stream['active'],
                'has_mask': camera_id in self.masks
            } for camera_id, stream in self.streams.items()]