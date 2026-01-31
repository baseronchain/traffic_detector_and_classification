import cv2
import time
import torch
from queue import Queue, Empty


class DetectorEngine:
    """Handles vehicle detection, tracking, and counting logic."""
    
    def __init__(self, model, device, vehicle_classes, confidence_threshold):
        self.model = model
        self.device = device
        self.vehicle_classes = vehicle_classes
        self.confidence_threshold = confidence_threshold
        
        # Tracking
        self.tracked_vehicles = {}
        self.counted_ids = set()
        
        # Counting parameters
        self.counting_line_y = 280
        self.line_offset = 40
        
        # Frame dimensions
        self.frame_width = 640
        self.frame_height = 480
        
        # Statistics
        self.total_frames = 0
        self.detection_count = 0
        
        # Frame queue for GUI
        self.frame_queue = Queue(maxsize=1)
        
        # Track call mode optimization
        self._track_call_mode = 0  # 0=unknown, 1=half+imgsz ok, 2=imgsz ok, 3=basic only
        self.infer_imgsz = 640
        self.use_half = (self.device == 'cuda')
        
        self.reset_counters()
    
    def reset_counters(self):
        """Reset all vehicle counters and tracking data."""
        self.vehicle_counts = {
            'mobil': 0,
            'motor': 0,
            'bus': 0,
            'truck': 0
        }
        self.total_vehicles = 0
        self.tracked_vehicles = {}
        self.counted_ids = set()
        self.total_frames = 0
        self.detection_count = 0
    
    def _track(self, frame):
        """Call model.track with a safe fallback for ultralytics version differences."""
        base_kwargs = dict(
            conf=self.confidence_threshold,
            iou=0.5,
            persist=True,
            verbose=False,
            classes=list(self.vehicle_classes.keys()),
            device=self.device,
        )

        if self._track_call_mode == 0:
            try:
                out = self.model.track(frame, imgsz=self.infer_imgsz, half=self.use_half, **base_kwargs)
                self._track_call_mode = 1
                return out
            except TypeError:
                try:
                    out = self.model.track(frame, imgsz=self.infer_imgsz, **base_kwargs)
                    self._track_call_mode = 2
                    return out
                except TypeError:
                    self._track_call_mode = 3
                    return self.model.track(frame, **base_kwargs)

        if self._track_call_mode == 1:
            return self.model.track(frame, imgsz=self.infer_imgsz, half=self.use_half, **base_kwargs)
        if self._track_call_mode == 2:
            return self.model.track(frame, imgsz=self.infer_imgsz, **base_kwargs)
        return self.model.track(frame, **base_kwargs)
    
    def push_frame(self, frame):
        """Push latest frame for GUI thread (drop old frames if GUI is slow)."""
        if frame is None:
            return
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put_nowait(frame)
        except Exception:
            # Never crash detector thread because GUI queue failed
            pass
    
    def detect_loop(self, cap, is_running_callback, update_stats_callback, update_fps_callback, root):
        """Main detection loop - GPU OPTIMIZED"""
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while is_running_callback() and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.total_frames += 1
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # YOLO TRACKING dengan GPU
                results = self._track(frame)
                
                # Draw counting zone
                zone_top = self.counting_line_y - self.line_offset
                zone_bottom = self.counting_line_y + self.line_offset
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, zone_top), (self.frame_width, zone_bottom), 
                            (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                
                cv2.line(frame, (0, self.counting_line_y), 
                        (self.frame_width, self.counting_line_y), 
                        (0, 0, 255), 3)
                cv2.putText(frame, "COUNTING ZONE", (10, zone_top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Process detections
                if results[0].boxes.id is not None:
                    self.detection_count += 1
                    boxes = results[0].boxes
                    track_ids = boxes.id.int().cpu().numpy()
                    classes = boxes.cls.int().cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    xyxys = boxes.xyxy.int().cpu().numpy()
                    
                    for track_id, class_id, conf, xyxy in zip(track_ids, classes, confs, xyxys):
                        if class_id in self.vehicle_classes:
                            vehicle_type = self.vehicle_classes[class_id]
                            
                            x1, y1, x2, y2 = map(int, xyxy)
                            centroid_x = (x1 + x2) // 2
                            centroid_y = (y1 + y2) // 2
                            
                            colors_map = {
                                'mobil': (52, 152, 219),
                                'motor': (155, 89, 182),
                                'bus': (230, 126, 34),
                                'truck': (231, 76, 60)
                            }
                            color = colors_map.get(vehicle_type, (255, 255, 255))
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.circle(frame, (centroid_x, centroid_y), 4, color, -1)
                            
                            label = f"ID:{track_id} {vehicle_type[:3]} {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                            
                            # COUNTING LOGIC
                            if track_id not in self.counted_ids:
                                in_zone = (zone_top <= centroid_y <= zone_bottom)
                                
                                if track_id in self.tracked_vehicles:
                                    prev_y = self.tracked_vehicles[track_id]['prev_y']
                                    prev_in_zone = self.tracked_vehicles[track_id].get('was_in_zone', False)
                                    
                                    crossed_down = (prev_y < zone_top and centroid_y > zone_bottom)
                                    crossed_up = (prev_y > zone_bottom and centroid_y < zone_top)
                                    entered_zone = (not prev_in_zone and in_zone)
                                    
                                    if crossed_down or crossed_up or entered_zone:
                                        self.vehicle_counts[vehicle_type] += 1
                                        self.counted_ids.add(track_id)
                                        
                                        cv2.putText(frame, "COUNTED!", (centroid_x - 40, centroid_y - 25),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        cv2.circle(frame, (centroid_x, centroid_y), 20, (0, 255, 0), 3)
                                        
                                        root.after(0, update_stats_callback)
                                
                                self.tracked_vehicles[track_id] = {
                                    'type': vehicle_type,
                                    'prev_y': centroid_y,
                                    'was_in_zone': in_zone
                                }
                
                # FPS calculation
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    detection_rate = (self.detection_count / self.total_frames * 100) if self.total_frames > 0 else 0
                    
                    # Color code FPS based on device
                    if self.device == 'cuda':
                        if fps >= 35:
                            fps_color = '#2ecc71'
                        elif fps >= 25:
                            fps_color = '#f39c12'
                        else:
                            fps_color = '#e74c3c'
                    else:
                        if fps >= 10:
                            fps_color = '#2ecc71'
                        elif fps >= 7:
                            fps_color = '#f39c12'
                        else:
                            fps_color = '#e74c3c'
                    
                    # GPU utilization (if CUDA)
                    gpu_mem = None
                    if self.device == 'cuda':
                        gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    
                    root.after(0, lambda: update_fps_callback(fps, fps_color, detection_rate, gpu_mem))
                    
                    fps_counter = 0
                    fps_start_time = time.time()
                
                self.push_frame(frame)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
