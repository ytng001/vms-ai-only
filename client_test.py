import requests
import time as time
def stop_all_cameras():
    print ("Stopping all cameras...")
    url = "http://localhost:8000/cameras/stop-all"
    
    try:
        response = requests.post(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('message')}")
            print(f"Cameras stopped: {result.get('count')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def start_all_cameras(server_url="http://localhost:8000"):
    """Start all cameras on the server"""
    url = f"{server_url}/cameras/start-all"
    try:
        response = requests.post(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
    
        result = response.json()
        print(f"Success: {result['message']}")
        print(f"Started {result.get('count', 0)} cameras")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error starting all cameras: {e}")
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json().get('detail', 'Unknown error')
                print(f"Detail: {error_detail}")
            except:
                print(f"Response: {e.response.text}")
        return False

def start_camera(camera_id, server_url="http://localhost:8000"):
    print(f"Starting all camera...")
    """Start a specific camera by ID"""
    url = f"{server_url}/cameras/{camera_id}/start"
    try:
        response = requests.post(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        result = response.json()
        print(f"Success: {result['message']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error starting camera {camera_id}: {e}")
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json().get('detail', 'Unknown error')
                print(f"Detail: {error_detail}")
            except:
                print(f"Response: {e.response.text}")
        return False

def list_cameras(server_url="http://localhost:8000"):
    """List all available cameras"""
    url = f"{server_url}/cameras"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        cameras = response.json().get('cameras', [])
        print(f"Found {len(cameras)} cameras:")
        for camera in cameras:
            active_status = "ACTIVE" if camera.get('active', False) else "STOPPED"
            print(f"  Camera {camera['id']}: {camera['source']} ({active_status}) - FPS: {camera.get('fps', 0):.1f}")
        
        return cameras
    except requests.exceptions.RequestException as e:
        print(f"Error listing cameras: {e}")
        return []

if __name__ == "__main__":
    stop_all_cameras()
    time.sleep(10)  # Wait for cameras to stop()
    start_all_cameras()