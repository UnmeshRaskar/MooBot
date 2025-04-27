import os
from PIL import Image, ImageOps
import glob

def preprocess_images(input_folder, output_folder, max_size=(300, 300), quality=70):
    """
    Preprocess cow images:
    1. Correct orientation using EXIF data
    2. Resize to smaller dimensions
    3. Optimize quality to reduce file size
    
    Args:
        input_folder: Folder containing original cow images
        output_folder: Folder to save processed images
        max_size: Maximum width and height (preserves aspect ratio)
        quality: JPEG quality (0-100), lower means smaller file size
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Get all jpg files in the input folder
    image_files = glob.glob(os.path.join(input_folder, "C*.jpg"))
    print("Found files:", image_files)

    
    for image_path in image_files:
        # Get the filename without path
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Open the image and fix orientation based on EXIF data
            with Image.open(image_path) as img:
                # Fix orientation using EXIF data
                img = ImageOps.exif_transpose(img)
                
                # Resize image while maintaining aspect ratio
                img.thumbnail(max_size, Image.LANCZOS)

                
                # Save the processed image with reduced quality
                img.save(output_path, "JPEG", quality=quality, optimize=True)
                
                # Print file size reduction
                original_size = os.path.getsize(image_path) / 1024  # KB
                new_size = os.path.getsize(output_path) / 1024  # KB
                reduction = (1 - (new_size / original_size)) * 100
                
                print(f"Processed {filename}: {original_size:.1f}KB → {new_size:.1f}KB ({reduction:.1f}% reduction)")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Set your input and output folders
    input_folder = r"D:\ECE\4th sem\WisEST\MooBot\code\cow_images"
    output_folder = r"D:\ECE\4th sem\WisEST\MooBot\code\cow_images_optimized"
    
    # Process images
    preprocess_images(
        input_folder=input_folder,
        output_folder=output_folder,
        max_size=(300, 300),  # Maximum width/height
        quality=70  # Moderate compression
    )
    
    print("Image preprocessing complete!")