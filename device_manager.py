import torch
from tkinter import messagebox
from ultralytics import YOLO


class DeviceManager:
    """Manages GPU/CPU detection and YOLO model loading."""
    
    @staticmethod
    def detect_device():
        """Sistem pngdeteksi, penghitung, dan klasifikasi kendaraan dengan YOLOv8l"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU DETECTED: {gpu_name}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return device
        else:
            device = 'cpu'
            print("")
            print("")
            return device
    
    @staticmethod
    def show_device_info(device):
        """Show device info popup"""
        if device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            message = f"Sistem pngdeteksi, penghitung, dan klasifikasi kendaraan dengan YOLOv8l\n\n"
            message += f"Device: {gpu_name}\n"
            message += f"VRAM: {vram:.1f} GB\n\n"
            message += f""
            message += f""
            message += f""
            message += f""
            messagebox.showinfo("GPU Ready!", message)
        else:
            message = "GPU NOT DETECTED\n\n"
            message += "Running on CPU mode (SLOW)\n\n"
            message += "To enable GPU:\n"
            message += "1. pip uninstall torch\n"
            message += "2. pip install torch --index-url \\\n"
            message += "   https://download.pytorch.org/whl/cu121\n"
            message += "3. Restart program"
            messagebox.showwarning("GPU Not Available", message)
    
    @staticmethod
    def load_model(device):
        """"""
        model_name = 'best.pt' if device == 'cuda' else 'best.pt'
        
        try:
            print(f"Loading {model_name} on {device}...")
            model = YOLO(model_name)
            
            # Force model ke device
            model.to(device)

            if device == 'cuda':
                try:
                    torch.backends.cudnn.benchmark = True
                    try:
                        torch.set_float32_matmul_precision('high')
                    except Exception:
                        pass
                except Exception:
                    pass

            # Fuse Conv+BN for slightly faster inference if available
            try:
                model.fuse()
            except Exception:
                pass
            
            print(f"Model loaded on {device}")
            
            if device == 'cuda':
                print(f"")
                print(f"")
            else:
                print(f"")
                print(f"")
            
            return model
                
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")
            raise
