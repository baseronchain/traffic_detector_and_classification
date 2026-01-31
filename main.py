import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

from device_manager import DeviceManager
from detector_engine import DetectorEngine
from gui_interface import GUIInterface


class TrafficDetectorGPU:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem pngdeteksi, penghitung, dan klasifikasi kendaraan dengan YOLOv8l")
        self.root.geometry("1200x750")
        
        # GPU Detection
        self.device = DeviceManager.detect_device()
        DeviceManager.show_device_info(self.device)
        
        # Load model dengan GPU
        self.model = DeviceManager.load_model(self.device)
        
        # kals kendaraan
        self.vehicle_classes = {
            2: 'mobil',
            3: 'motor',
            5: 'bus',
            7: 'truck'
        }
        
        # Initialize detector engine
        self.detector_engine = DetectorEngine(
            model=self.model,
            device=self.device,
            vehicle_classes=self.vehicle_classes,
            confidence_threshold=0.5
        )
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.video_thread = None
        
        # Initialize GUI
        self.gui = GUIInterface(self.root, self.device, self.detector_engine)
        self.gui.setup_gui(
            select_video_callback=self.select_video,
            stop_video_callback=self.stop_video,
            reset_all_callback=self.reset_all,
            update_line_position_callback=self.update_line_position,
            update_confidence_callback=self.update_confidence
        )
        
        # Start GUI update loop
        self.root.after(self.gui._gui_interval_ms, self.gui.gui_update_loop)
    
    def update_line_position(self, value):
        """Update counting line position."""
        self.detector_engine.counting_line_y = int(float(value))
    
    def update_confidence(self, value):
        """Update confidence threshold."""
        self.detector_engine.confidence_threshold = float(value)
        percentage = int(self.detector_engine.confidence_threshold * 100)
        self.gui.conf_label.config(text=f"{percentage}%")
    
    def select_video(self):
        """Open file dialog to select video."""
        file_path = filedialog.askopenfilename(
            title="Pilih Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.start_detection(file_path)
    
    def start_detection(self, source):
        """Start video detection."""
        if self.is_running:
            self.stop_video()
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat membuka video")
            return
        
        self.is_running = True
        self.detector_engine.reset_counters()
        self.update_stats()
        
        self.video_thread = threading.Thread(target=self._detect_thread, daemon=True)
        self.video_thread.start()
        
        self.gui.update_status(f"Status: Processing | Device: {self.device.upper()} ")
    
    def _detect_thread(self):
        """Thread wrapper for detection loop."""
        self.detector_engine.detect_loop(
            cap=self.cap,
            is_running_callback=lambda: self.is_running,
            update_stats_callback=self.update_stats,
            update_fps_callback=self.update_fps,
            root=self.root
        )
        # Wloop ends, stopp video
        self.stop_video()
    
    def update_stats(self):
        """Update GUI statistics."""
        self.gui.update_stats(self.detector_engine.vehicle_counts)
    
    def update_fps(self, fps, fps_color, detection_rate, gpu_mem):
        """Update FPS display."""
        self.gui.update_fps_display(fps, fps_color, detection_rate, gpu_mem)
    
    def stop_video(self):
        """Stop video processing."""
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        # Tkinter is not thread-safe; always update widgets via main thread
        try:
            self.root.after(0, lambda: self.gui.update_status("Status: Stopped"))
        except Exception:
            pass
    
    def reset_all(self):
        """Reset all counters and UI."""
        self.stop_video()
        self.detector_engine.reset_counters()
        self.update_stats()
        self.gui.clear_video_display()
        self.gui.reset_performance_display()
        self.gui.update_status(f"Status: Ready | Device: {self.device.upper()}")
    
    def on_closing(self):
        """Handle window closing."""
        self.stop_video()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TrafficDetectorGPU(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
