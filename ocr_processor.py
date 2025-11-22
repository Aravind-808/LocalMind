import pytesseract
import cv2
import numpy as np
from PIL import Image
import os

class OCRProcessor:
    @staticmethod
    def preprocess_image(image_path):
        """
        Loads an image, converts to grayscale, and applies thresholding
        to make text pop out for Tesseract.
        """
        # Read image using OpenCV
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply binary thresholding (Otsu's method) helps separate text from background
        # This creates a crisp black and white image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    @staticmethod
    def extract_text(image_path):
        """
        Orchestrates the OCR process.
        """
        try:
            # Pre-process for better accuracy
            processed_img_matrix = OCRProcessor.preprocess_image(image_path)
            
            # Convert back to PIL image for Tesseract
            pil_img = Image.fromarray(processed_img_matrix)
            
            # Run Tesseract
            # --psm 3 is default (Auto page segmentation, but no OSD)
            text = pytesseract.image_to_string(pil_img, config='--psm 3')
            
            return text.strip()
        except Exception as e:
            print(f"OCR Failed for {image_path}: {e}")
            return ""