import cv2
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from glob import glob

class BatchImageClassifier:
    def __init__(self, model_path, conf_threshold=0.25):
        """Initialize the classifier with model path and confidence threshold."""
        self.conf_threshold = conf_threshold
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocess_image(self, img):
        """Preprocess image for model input."""
        face = cv2.resize(img, (96, 96))
        h = face.shape[0]
        
        # Extract regions
        upper_h = int(h * 0.3)
        middle_h = int(h * 0.3)
        
        # Extract and resize regions
        upper_region = cv2.resize(face[:upper_h, :], (32, 32))
        middle_region = cv2.resize(face[upper_h:upper_h+middle_h, :], (32, 32))
        lower_region = cv2.resize(face[upper_h+middle_h:, :], (32, 32))
        
        # Normalize
        upper_region = np.expand_dims(upper_region, axis=0) / 255.0
        middle_region = np.expand_dims(middle_region, axis=0) / 255.0
        lower_region = np.expand_dims(lower_region, axis=0) / 255.0
        
        return [upper_region, middle_region, lower_region]

    def process_face(self, frame, face_box):
        """Process a single face region."""
        x, y, w, h = face_box
        
        # Extract and preprocess face region
        face_img = frame[y:y+h, x:x+w]
        input_tensors = self.preprocess_image(face_img)
        
        # Run inference
        predictions = self.model.predict(input_tensors, verbose=0)
        
        # Get class and confidence
        class_idx, confidence = self.postprocess_predictions(predictions)
        
        return class_idx, confidence

    def detect_faces(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def postprocess_predictions(self, predictions):
        """Process model predictions to get class and confidence."""
        pred_scores = predictions[0]
        pred_class_idx = np.argmax(pred_scores)
        confidence = pred_scores[pred_class_idx]
        
        if confidence >= self.conf_threshold:
            return pred_class_idx, confidence
        return None, None

    def draw_prediction(self, img, class_idx, confidence, face_box):
        """Draw prediction label and bounding box on the image."""
        if class_idx is None:
            return img
            
        # Get color for current class
        color = self.colors[class_idx].astype(int).tolist()
        
        # Draw bounding box
        x, y, w, h = face_box
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        # Prepare label
        class_name = self.class_names[class_idx]
        label = f'{class_name}: {confidence:.2f}'
        
        # Calculate label position (above the box)
        font_scale = 0.6
        thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(img, 
                     (x, y - label_h - 10), 
                     (x + label_w, y), 
                     color, 
                     -1)
        
        # Draw label text
        cv2.putText(img, 
                    label, 
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (255, 255, 255), 
                    thickness)
        
        return img

    def process_image_folder(self, input_folder, output_folder):
        """Process all images in a folder and save results."""
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get list of image files
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(glob(os.path.join(input_folder, ext)))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Read image
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                # Get base filename
                base_name = os.path.basename(img_path)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # If no faces found, save this information
                if len(faces) == 0:
                    results.append({
                        'image': base_name,
                        'faces_detected': 0,
                        'emotions': []
                    })
                    continue
                
                # Process each detected face
                face_results = []
                for face_box in faces:
                    # Get predictions for this face
                    class_idx, confidence = self.process_face(frame, face_box)
                    
                    # Draw predictions
                    if class_idx is not None:
                        frame = self.draw_prediction(frame, class_idx, confidence, face_box)
                        face_results.append({
                            'emotion': self.class_names[class_idx],
                            'confidence': float(confidence)
                        })
                
                # Save results
                results.append({
                    'image': base_name,
                    'faces_detected': len(faces),
                    'emotions': face_results
                })
                
                # Save annotated image
                output_path = os.path.join(output_folder, base_name)
                cv2.imwrite(output_path, frame)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        # Save results to file
        import json
        results_path = os.path.join(output_folder, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Processed images saved to: {output_folder}")
        print(f"Analysis results saved to: {results_path}")

if __name__ == "__main__":
    MODEL_PATH = "emotion_region_classifier.h5"
    CONF_THRESHOLD = 0.5
    
    # Specify input and output folders
    INPUT_FOLDER = "input_images"  # Replace with your input folder path
    OUTPUT_FOLDER = "analyzed_images"  # Replace with your desired output folder path
    
    classifier = BatchImageClassifier(MODEL_PATH, CONF_THRESHOLD)
    classifier.process_image_folder(INPUT_FOLDER, OUTPUT_FOLDER)