import cv2
import numpy as np
import tensorflow as tf
import time

class RealTimeClassifier:
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
        input_img = cv2.resize(img, (96, 96))
        input_img = np.expand_dims(input_img, axis=0)
        return input_img

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

    def process_face(self, frame, face_box):
        """Process a single face region."""
        x, y, w, h = face_box
        
        # Extract and preprocess face region
        face_img = frame[y:y+h, x:x+w]
        input_tensor = self.preprocess_image(face_img)
        
        # Run inference
        predictions = self.model.predict(input_tensor, verbose=0)
        
        # Get class and confidence
        class_idx, confidence = self.postprocess_predictions(predictions)
        
        return class_idx, confidence

    def run_classification(self):
        """Run real-time classification using webcam feed."""
        cap = cv2.VideoCapture(2)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Detect faces
                start_time = time.time()
                faces = self.detect_faces(frame)
                
                # Process each detected face
                for face_box in faces:
                    # Get predictions for this face
                    class_idx, confidence = self.process_face(frame, face_box)
                    
                    # Draw predictions
                    if class_idx is not None:
                        frame = self.draw_prediction(frame, class_idx, confidence, face_box)
                
                # Calculate and draw FPS
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(frame, 
                           f'FPS: {fps:.2f}',
                           (20, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1, 
                           (0, 255, 0), 
                           2)
                
                # Show framef
                cv2.imshow('Classification', frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("Stopped by User")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "simpsons_classifier.h5"
    CONF_THRESHOLD = 0.25
    
    classifier = RealTimeClassifier(MODEL_PATH, CONF_THRESHOLD)
    classifier.run_classification()