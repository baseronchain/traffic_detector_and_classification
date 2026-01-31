import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from queue import Empty
import torch


class GUIInterface:
    """Manages all GUI components and rendering."""
    
    def __init__(self, root, device, detector_engine):
        self.root = root
        self.device = device
        self.detector_engine = detector_engine
        
        # GUI rendering throttling
        self.target_display_fps = 20
        self._gui_interval_ms = max(1, int(1000 / self.target_display_fps))
        
        # References to GUI elements (will be set in setup_gui)
        self.video_label = None
        self.total_label = None
        self.detail_labels = {}
        self.percentage_labels = {}
        self.fps_label = None
        self.detect_rate_label = None
        self.gpu_util_label = None
        self.status_label = None
        self.conf_label = None
        self.line_scale = None
        self.conf_scale = None
    
    def setup_gui(self, select_video_callback, stop_video_callback, reset_all_callback,
                  update_line_position_callback, update_confidence_callback):
        """Setup GUI"""
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#1a1a1a', padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Device indicator
        device_color = '#2ecc71' if self.device == 'cuda' else '#e74c3c'
        device_text = f"GPU MODE" if self.device == 'cuda' else "CPU MODE"
        tk.Label(control_frame, text=device_text, bg='#1a1a1a', 
                fg=device_color, font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=10)
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0).split()[-2:]  # Get model name
            tk.Label(control_frame, text=f"({' '.join(gpu_name)})", 
                    bg='#1a1a1a', fg='white', font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(control_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        # Buttons
        button_frame = tk.Frame(control_frame, bg='#1a1a1a')
        button_frame.pack(side=tk.LEFT)
        
        tk.Button(button_frame, text="Pilih Video", command=select_video_callback,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Stop", command=stop_video_callback,
                 bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Reset", command=reset_all_callback,
                 bg='#f39c12', fg='white', font=('Arial', 10, 'bold'),
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        # Adjustments
        ttk.Separator(control_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        adjust_frame = tk.Frame(control_frame, bg='#1a1a1a')
        adjust_frame.pack(side=tk.LEFT)
        
        tk.Label(adjust_frame, text="Posisi Garis:", bg='#1a1a1a', fg='white',
                font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.line_scale = tk.Scale(adjust_frame, from_=100, to=380, orient=tk.HORIZONTAL,
                                   command=update_line_position_callback, bg='#34495e', fg='white',
                                   length=120, highlightthickness=0)
        self.line_scale.set(self.detector_engine.counting_line_y)
        self.line_scale.pack(side=tk.LEFT, padx=5)
        
        tk.Label(adjust_frame, text="Confidence:", bg='#1a1a1a', fg='white',
                font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(15, 5))
        
        self.conf_scale = tk.Scale(adjust_frame, from_=0.3, to=0.7, resolution=0.05,
                                   orient=tk.HORIZONTAL, command=update_confidence_callback,
                                   bg='#34495e', fg='white', length=120, highlightthickness=0)
        self.conf_scale.set(self.detector_engine.confidence_threshold)
        self.conf_scale.pack(side=tk.LEFT, padx=5)
        
        self.conf_label = tk.Label(adjust_frame, text="50%", bg='#1a1a1a', fg='#2ecc71',
                                   font=('Arial', 9, 'bold'), width=5)
        self.conf_label.pack(side=tk.LEFT, padx=5)
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame
        video_frame = tk.LabelFrame(main_frame, text="Video Smpel", 
                                   font=('Arial', 12, 'bold'), padx=5, pady=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Stats frame
        stats_frame = tk.LabelFrame(main_frame, text="Statistik Kendaraan", 
                                   font=('Arial', 12, 'bold'), padx=10, pady=10, width=300)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        stats_frame.pack_propagate(False)
        
        # Total
        total_frame = tk.Frame(stats_frame, bg='#34495e', padx=10, pady=10)
        total_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(total_frame, text="TOTAL LEWAT", 
                font=('Arial', 10, 'bold'), bg='#34495e', fg='white').pack()
        self.total_label = tk.Label(total_frame, text="0", 
                                   font=('Arial', 36, 'bold'), bg='#34495e', fg='#2ecc71')
        self.total_label.pack()
        
        # Vehicle details
        self.detail_labels = {}
        self.percentage_labels = {}
        
        vehicle_icons = {'mobil': '', 'motor': '', 'bus': '', 'truck': ''}
        colors = {'mobil': '#3498db', 'motor': '#9b59b6', 'bus': '#e67e22', 'truck': '#e74c3c'}
        
        for vehicle, icon in vehicle_icons.items():
            frame = tk.Frame(stats_frame, bg=colors[vehicle], padx=10, pady=8)
            frame.pack(fill=tk.X, pady=4)
            
            tk.Label(frame, text=f"{icon} {vehicle.upper()}", 
                    font=('Arial', 10, 'bold'), bg=colors[vehicle], 
                    fg='white').pack(anchor='w')
            
            count_frame = tk.Frame(frame, bg=colors[vehicle])
            count_frame.pack(fill=tk.X)
            
            self.detail_labels[vehicle] = tk.Label(count_frame, text="0", 
                                                   font=('Arial', 18, 'bold'), 
                                                   bg=colors[vehicle], fg='white')
            self.detail_labels[vehicle].pack(side=tk.LEFT)
            
            self.percentage_labels[vehicle] = tk.Label(count_frame, text="(0%)", 
                                                       font=('Arial', 9), 
                                                       bg=colors[vehicle], fg='white')
            self.percentage_labels[vehicle].pack(side=tk.RIGHT, padx=10)
        
        # Performance
        perf_frame = tk.Frame(stats_frame, bg='#ecf0f1', padx=10, pady=8)
        perf_frame.pack(fill=tk.X, pady=(15, 0))
        
        tk.Label(perf_frame, text="", font=('Arial', 9, 'bold'), 
                bg='#ecf0f1').pack(anchor='w')
        
        self.fps_label = tk.Label(perf_frame, text="", 
                                 font=('Arial', 8, 'bold'), bg='#ecf0f1', anchor='w')
        self.fps_label.pack(fill=tk.X)
        
        self.detect_rate_label = tk.Label(perf_frame, text="", 
                                          font=('Arial', 8), bg='#ecf0f1', anchor='w')
        self.detect_rate_label.pack(fill=tk.X)
        
        if self.device == 'cuda':
            self.gpu_util_label = tk.Label(perf_frame, text="", 
                                          font=('Arial', 8), bg='#ecf0f1', anchor='w')
            self.gpu_util_label.pack(fill=tk.X)
        
        # Status
        status_bg = '#2ecc71' if self.device == 'cuda' else '#e74c3c'
        self.status_label = tk.Label(self.root, 
                                    text=f"Status: Ready | Device: {self.device.upper()}", 
                                    font=('Arial', 9, 'bold'), bg=status_bg, fg='white',
                                    anchor='w', padx=10, pady=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def gui_update_loop(self):
        """Render loop on Tk main thread (throttled)."""
        try:
            frame = None
            try:
                frame = self.detector_engine.frame_queue.get_nowait()
            except Empty:
                frame = None

            if frame is not None:
                self.display_frame(frame)

            if self.root.winfo_exists():
                self.root.after(self._gui_interval_ms, self.gui_update_loop)
        except tk.TclError:
            return
    
    def display_frame(self, frame):
        """Display a frame in the video label."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk
    
    def update_stats(self, vehicle_counts):
        """Update statistics display."""
        total_vehicles = sum(vehicle_counts.values())
        self.total_label.config(text=str(total_vehicles))
        
        for vehicle, count in vehicle_counts.items():
            self.detail_labels[vehicle].config(text=str(count))
            
            if total_vehicles > 0:
                percentage = (count / total_vehicles) * 100
                self.percentage_labels[vehicle].config(text=f"({percentage:.1f}%)")
            else:
                self.percentage_labels[vehicle].config(text="(0%)")
    
    def update_fps_display(self, fps, fps_color, detection_rate, gpu_mem):
        """Update FPS and performance metrics."""
        self.fps_label.config(text=f"FPS: {fps:.1f}", fg=fps_color)
        self.detect_rate_label.config(text=f"")
        
        if self.device == 'cuda' and gpu_mem is not None:
            self.gpu_util_label.config(text=f"")
    
    def update_status(self, text):
        """Update status label."""
        self.status_label.config(text=text)
    
    def clear_video_display(self):
        """Clear the video display."""
        self.video_label.config(image='')
    
    def reset_performance_display(self):
        """Reset performance metrics display."""
        self.fps_label.config(text="FPS: --")
        self.detect_rate_label.config(text="Detection: --")
