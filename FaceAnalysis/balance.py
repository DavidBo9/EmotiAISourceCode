from glob import glob
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

def convert_png_to_jpg(input_dir):
    """Convert all PNG files in the directory to JPG format."""
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return
        
    png_files = glob(os.path.join(input_dir, "**/*.png"), recursive=True)
    print(f"Found {len(png_files)} PNG files to convert")
    
    for png_file in tqdm(png_files, desc="Converting PNG to JPG"):
        try:
            # Read PNG image
            image = cv2.imread(png_file)
            if image is None:
                print(f"Failed to read: {png_file}")
                continue
                
            # Create new JPG filename
            jpg_file = os.path.splitext(png_file)[0] + '.jpg'
            
            # Save as JPG
            cv2.imwrite(jpg_file, image)
            
            # Remove original PNG file
            os.remove(png_file)
            
        except Exception as e:
            print(f"Error converting {png_file}: {str(e)}")

def augment_class(input_dir, class_name, target_count):
    """
    Augment images in a specific class directory until reaching target count,
    using all source images more evenly.
    """
    IMG_SIZE = (96,96)
    class_dir = os.path.join(input_dir, class_name)
    
    # Check if directory exists
    if not os.path.exists(class_dir):
        print(f"Error: Directory not found: {class_dir}")
        return
    
    # Convert any PNG files to JPG first
    convert_png_to_jpg(class_dir)
    
    # Now look for both JPG and PNG files (in case conversion failed)
    existing_files = glob(os.path.join(class_dir, "*.jpg")) + glob(os.path.join(class_dir, "*.png"))
    current_count = len(existing_files)
    
    # Check if we found any files
    if current_count == 0:
        print(f"Error: No images found in {class_dir}")
        return
        
    print(f"Found {current_count} existing images in {class_name} class")
    
    if current_count >= target_count:
        print(f"{class_name} already has sufficient images ({current_count} images)")
        return
    
    # Calculate how many new images we need
    num_to_generate = target_count - current_count
    
    # Calculate how many augmented images to generate per original image
    augmentations_per_image = int(np.ceil(num_to_generate / current_count))
    
    print(f"Will generate {augmentations_per_image} augmented images per source image")
    
    # Create augmentation pipeline with more variety
    datagen = ImageDataGenerator(
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        channel_shift_range=20  # Added subtle color shifts
    )
    
    # Generate new images
    generated_count = 0
    pbar = tqdm(total=num_to_generate, desc=f"Augmenting {class_name}")
    
    # Shuffle the existing files to ensure random selection
    np.random.shuffle(existing_files)
    
    # Keep cycling through all source images until we reach target count
    while generated_count < num_to_generate:
        for source_image_path in existing_files:
            if generated_count >= num_to_generate:
                break
                
            try:
                # Read and preprocess source image
                image = cv2.imread(source_image_path)
                if image is None:
                    print(f"Failed to read: {source_image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMG_SIZE)
                image = image.reshape((1,) + image.shape)
                
                # Generate augmentations for this source image
                for _ in range(min(augmentations_per_image, num_to_generate - generated_count)):
                    for batch in datagen.flow(image, batch_size=1):
                        augmented_image = batch[0]
                        # Create unique filename using both count and source image name
                        source_basename = os.path.splitext(os.path.basename(source_image_path))[0]
                        new_filename = os.path.join(
                            class_dir, 
                            f"aug_{source_basename}_{generated_count}_{class_name}.jpg"
                        )
                        
                        augmented_image = cv2.cvtColor(augmented_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(new_filename, augmented_image)
                        
                        generated_count += 1
                        pbar.update(1)
                        break
                        
            except Exception as e:
                print(f"Error processing {source_image_path}: {str(e)}")
                continue
    
    pbar.close()
    final_count = len(glob(os.path.join(class_dir, "*.jpg")) + glob(os.path.join(class_dir, "*.png")))
    print(f"Finished augmenting {class_name}: {num_to_generate} images generated")
    print(f"Final count for {class_name}: {final_count} images")

# Use the function to augment classes
emotions = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
target = 5126  # Your target count

print("Starting augmentation process...")

# First verify all directories exist
base_dir = 'trainingdataset'
if not os.path.exists(base_dir):
    print(f"Error: Base directory not found: {base_dir}")
else:
    for emotion in emotions:
        emotion_dir = os.path.join(base_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found for {emotion}: {emotion_dir}")
    
    # Proceed with augmentation
    for emotion in emotions:
        print(f"\nProcessing {emotion}...")
        augment_class(base_dir, emotion, target)