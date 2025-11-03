"""
Camera capture thread for real-time spectral data acquisition.
Handles video capture in a separate thread to prevent UI blocking.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import numpy as np


class CameraThread(QThread):
    """Thread for continuous camera frame capture."""
    
    # Signals
    frame_ready = pyqtSignal(np.ndarray)  # Emits captured frame
    error_occurred = pyqtSignal(str)  # Emits error messages

    def __init__(self, device_id=0, width=1280, height=720, fps=30):
        super().__init__()
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.cap = None
        
    def init_camera(self):
        """Initialize OpenCV USB camera."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Using OpenCV camera: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Camera initialization failed: {str(e)}")
            return False
    
    def run(self):
        """Main thread loop for capturing frames."""
        if not self.init_camera():
            self.error_occurred.emit("Failed to initialize camera")
            return
        
        self.running = True
        
        while self.running:
            try:
                # Capture from OpenCV
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame_ready.emit(frame)
                else:
                    self.error_occurred.emit("Failed to read frame")
                    break
                
                # Small delay to prevent excessive CPU usage
                self.msleep(10)
                
            except Exception as e:
                self.error_occurred.emit(f"Frame capture error: {str(e)}")
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release camera resources."""
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
    
    def stop(self):
        """Stop the capture thread."""
        self.running = False
        self.wait()  # Wait for thread to finish
    
    def set_camera_exposure(self, exposure):
        """Set camera exposure."""
        if self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                return True
            except Exception as e:
                print(f"Failed to set exposure: {e}")
                return False
        return False
    
    def set_camera_brightness(self, brightness):
        """Set camera brightness."""
        if self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                return True
            except:
                return False
        return False
    
    def set_camera_contrast(self, contrast):
        """Set camera contrast."""
        if self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
                return True
            except:
                return False
        return False
