from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2 as cv
import numpy as np

def select_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Get screen dimensions before destroying window
    try:
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
    except:
        screen_width = 1920
        screen_height = 1080
    
    filepath = askopenfilename(
        title="Select image file",
        filetypes=[
            ("Image files", "*.jpeg *.jpg *.png *.tiff *.webp *.bmp *.gif *.ico *.dds *.exr *.hdr *.jp2 *.pbm *.pgm *.ppm *.sr *.ras"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    
    if not filepath:
        raise FileNotFoundError("No file selected.")
    
    # Reserve space for window decorations and taskbar
    max_width = int(screen_width * 0.8)
    max_height = int(screen_height * 0.8)
    
    # Try OpenCV first
    img = cv.imread(filepath, cv.IMREAD_COLOR)
    
    # If OpenCV fails, try PIL with more format support
    if img is None:
        try:
            pil_img = Image.open(filepath)
            # Handle all PIL modes
            if pil_img.mode == 'RGBA':
                # Convert RGBA to RGB with white background
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[-1])
                pil_img = background
            elif pil_img.mode in ['P', 'LA', 'L']:
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode not in ['RGB', 'BGR']:
                pil_img = pil_img.convert('RGB')
            
            img = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    # Resize if image is larger than screen - maintain aspect ratio
    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        # Calculate scale to fit both dimensions while preserving aspect ratio
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)  # Use smaller scale to fit both dimensions
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use INTER_AREA for downscaling (best quality, preserves aspect ratio)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    return img

def save_image(image):
    
    root = Tk()
    root.withdraw()  
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        initialfile= "",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
    )
    root.update()
    root.destroy()

    if file_path:
        cv.imwrite(file_path, image)
        return file_path
    return None

def reset_trackbars(window_name: str, defaults: dict):
    """
    window_name : str
        Name of the window where the trackbars exist.
    defaults : dict
        Dictionary {trackbar_name: default_value}
    """
    for name, value in defaults.items():
        cv.setTrackbarPos(name, window_name, value)

def help_menu(img, show_help=False):
    if not show_help:
        cv.putText(img, "Press 'h' for help", (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)
        return

    lines = [
        "HELP MENU:",
        "A - Auto Mode",
        "R - Reset",
        "S - Save",
        "ESC - Exit"
    ]
    y0 = 20
    for i, line in enumerate(lines):
        y = y0 + i * 25
        cv.putText(img, line, (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
