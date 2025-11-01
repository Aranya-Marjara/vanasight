import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import os
from torchvision import models, transforms
import requests
from io import BytesIO
import logging
from typing import List, Tuple, Optional
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanasight")

class VanaSight:
    """End-to-end computer vision pipeline from image loading to AI generation."""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
        )
        self.classification_model = None
        self.classification_categories = None
        logger.info(f"Initialized VanaSight on device: {self.device}")
    
    def load_image(self, source: str) -> np.ndarray:
        """Load image from file path or URL."""
        try:
            if os.path.exists(source):
                image = cv2.imread(source)
                if image is None:
                    raise ValueError(f"Could not load image from {source}")
            elif source.startswith(('http://', 'https://')):
                response = requests.get(source, timeout=10)
                response.raise_for_status()
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(source)
                if image is None:
                    raise FileNotFoundError(f"Image not found: {source}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Loaded image with shape: {image_rgb.shape}")
            return image_rgb
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            synthetic = self._generate_synthetic_image()
            logger.info("Using synthetic image as fallback")
            return synthetic
    
    def _generate_synthetic_image(self) -> np.ndarray:
        """Generate a fallback synthetic image."""
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (255, 0, 0), -1)
        cv2.circle(image, (400, 200), 80, (0, 255, 0), -1)
        cv2.line(image, (50, 400), (550, 400), (0, 0, 255), 5)
        return image
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement pipeline."""
        logger.info("Applying image enhancement pipeline...")
        enhanced = image.copy().astype(np.float32)
        
        # Denoising
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced.astype(np.uint8), None, 10, 10, 7, 21
        ).astype(np.float32)
        
        # Contrast enhancement with CLAHE
        lab = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        
        # Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Color enhancement
        hsv = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Gamma correction
        gamma = 0.9
        enhanced = np.clip(255.0 * (enhanced / 255.0) ** (1.0 / gamma), 0, 255)
        
        logger.info("Image enhancement complete")
        return enhanced.astype(np.uint8)
    
    def _load_classification_model(self):
        """Load classification model and labels."""
        if self.classification_model is None:
            logger.info("Loading ResNet-18 classification model...")
        
        # Updated to use weights instead of pretrained (fixes deprecation warnings)
            from torchvision.models import ResNet18_Weights
            self.classification_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
            self.classification_model.eval()
            if torch.cuda.is_available():
                self.classification_model.to(self.device)
            self._load_imagenet_labels()

    def _load_imagenet_labels(self):
        """Load ImageNet class labels."""
        labels_path = "imagenet_classes.txt"
        if not os.path.exists(labels_path):
            logger.info("Downloading ImageNet labels...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
                labels_path
            )
        
        with open(labels_path, "r") as f:
            self.classification_categories = [s.strip() for s in f.readlines()]
    
    def classify_image(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Classify image using pre-trained ResNet."""
        logger.info("Running image classification...")
        
        try:
            self._load_classification_model()
            
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            pil_image = Image.fromarray(image)
            input_tensor = preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.classification_model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            results = []
            for i in range(top_k):
                class_name = self.classification_categories[top_indices[i]]
                probability = top_probs[i].item()
                results.append((class_name, probability))
                logger.info(f"  {class_name}: {probability:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return [("classification_error", 0.0)]
    
    def detect_objects(self, image: np.ndarray) -> np.ndarray:
        """Perform object detection using contour-based approach."""
        logger.info("Running object detection...")
        
        detection_image = image.copy()
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = max(image.shape[0] * image.shape[1] * 0.001, 500)
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(detection_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(detection_image, "Object", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            logger.info(f"Detected {len(valid_contours)} objects")
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            cv2.putText(detection_image, "Detection Failed", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return detection_image
    
    def generate_ai_art(self, image: np.ndarray, 
                       classification_results: List[Tuple[str, float]]) -> np.ndarray:
        """Generate AI-inspired artistic version of the input image."""
        logger.info("Generating AI-inspired artistic version...")
        
        try:
            top_class = classification_results[0][0] if classification_results else "abstract"
            pil_image = Image.fromarray(image)
            
            if any(word in top_class.lower() for word in ['landscape', 'outdoor', 'nature']):
                artistic = pil_image.filter(ImageFilter.GaussianBlur(0.5))
                artistic = ImageEnhance.Color(artistic).enhance(1.4)
                artistic = ImageEnhance.Contrast(artistic).enhance(1.2)
            elif any(word in top_class.lower() for word in ['portrait', 'person', 'face']):
                artistic = pil_image.filter(ImageFilter.SMOOTH)
                artistic = ImageEnhance.Color(artistic).enhance(1.1)
                artistic = ImageEnhance.Brightness(artistic).enhance(1.1)
            else:
                artistic = pil_image.filter(ImageFilter.EDGE_ENHANCE)
                artistic = ImageEnhance.Color(artistic).enhance(1.3)
                artistic = ImageEnhance.Sharpness(artistic).enhance(1.5)
            
            np_artistic = np.array(artistic)
            noise = np.random.randint(-20, 20, np_artistic.shape, dtype=np.int16)
            np_artistic = np.clip(np_artistic.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            np_artistic = self._apply_vignette(np_artistic)
            
            logger.info("AI art generation complete")
            return np_artistic
            
        except Exception as e:
            logger.error(f"AI art generation failed: {e}")
            return image
    
    def _apply_vignette(self, image: np.ndarray, intensity: float = 0.8) -> np.ndarray:
        """Apply vignette effect to image."""
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 3)
        kernel_y = cv2.getGaussianKernel(rows, rows / 3)
        kernel = kernel_y * kernel_x.T
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
        mask = 1.0 - kernel * intensity
        
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def run_pipeline(self, image_source: str, 
                    output_path: str = "vanasight_results.jpg") -> dict:
        """Run complete vision pipeline on input image."""
        logger.info("Starting VanaSight pipeline...")
        
        original = self.load_image(image_source)
        enhanced = self.enhance_image(original)
        classification = self.classify_image(enhanced)
        detection = self.detect_objects(enhanced)
        ai_art = self.generate_ai_art(enhanced, classification)
        
        self._visualize_results(original, enhanced, detection, ai_art, output_path)
        
        results = {
            'original': original,
            'enhanced': enhanced,
            'classification': classification,
            'detection': detection,
            'ai_art': ai_art,
            'output_path': output_path
        }
        
        logger.info("VanaSight pipeline completed successfully!")
        return results
    
    def _visualize_results(self, original: np.ndarray, enhanced: np.ndarray,
                          detection: np.ndarray, ai_art: np.ndarray,
                          output_path: str):
        """Create and save visualization of all pipeline steps."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('VanaSight: From Pixels to Perception', fontsize=16, fontweight='bold')
        
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('1. Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(enhanced)
        axes[0, 1].set_title('2. Enhanced (OpenCV)', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(detection)
        axes[1, 0].set_title('3. Object Detection', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ai_art)
        axes[1, 1].set_title('4. AI Reimagined', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Results saved to: {output_path}")
        plt.close()

def main():
    """Command-line interface for VanaSight."""
    parser = argparse.ArgumentParser(
        description="VanaSight: From Pixels to Perception - Complete CV Pipeline"
    )
    parser.add_argument('--input', '-i', required=True, help='Input image path or URL')
    parser.add_argument('--output', '-o', default='vanasight_results.jpg', 
                       help='Output path for results')
    parser.add_argument('--device', '-d', choices=['auto', 'cpu', 'cuda'],
                       default='auto', help='Device to run models on')
    
    args = parser.parse_args()
    
    pipeline = VanaSight(device=args.device)
    results = pipeline.run_pipeline(args.input, args.output)
    
    print("\n" + "="*60)
    print("VANASIGHT PIPELINE COMPLETED!")
    print("="*60)
    print(f"📊 Results saved to: {results['output_path']}")
    if results['classification']:
        top_class, confidence = results['classification'][0]
        print(f"🔍 Top classification: {top_class} ({confidence:.2%})")
    print("="*60)

if __name__ == "__main__":
    main()
