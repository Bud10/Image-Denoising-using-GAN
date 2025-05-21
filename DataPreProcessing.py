import tensorflow as tf
import cv2
import numpy as np
import os
import glob
import shutil

def load_and_preprocess(noisy_path, clean_path):
    noisy_image = cv2.imread(noisy_path)
    clean_image = cv2.imread(clean_path)
    if noisy_image is None or clean_image is None:
        raise ValueError(f"Failed to load images: {noisy_path}, {clean_path}")
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    noisy_image = cv2.resize(noisy_image, (256, 256))
    clean_image = cv2.resize(clean_image, (256, 256))
    noisy_image = noisy_image.astype(np.float32)
    clean_image = clean_image.astype(np.float32)
    noisy_image = (noisy_image - 127.5) / 127.5  # Normalize to [-1, 1]
    clean_image = (clean_image - 127.5) / 127.5  # Normalize to [-1, 1]
    return noisy_image, clean_image

def prepare_data(noisy_dirs, clean_dir, cache_dir="D:/dataset/cache"):
    # Clear existing cache directory to ensure overwrite
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    clean_images = {}
    valid_extensions = (".jpg", ".png", ".jpeg")
    
    for celeb_folder in os.listdir(clean_dir):
        celeb_path = os.path.join(clean_dir, celeb_folder)
        if not os.path.isdir(celeb_path):
            print(f"Skipping non-directory: {celeb_path}")
            continue
        
        clean_paths = sorted(
            glob.glob(os.path.join(celeb_path, "*.jpg")) +
            glob.glob(os.path.join(celeb_path, "*.png")) +
            glob.glob(os.path.join(celeb_path, "*.jpeg"))
        )
        
        for clean_path in clean_paths:
            clean_name = os.path.splitext(os.path.basename(clean_path))[0]
            clean_images.setdefault(celeb_folder, {})[clean_name] = clean_path
    
    # Collect noisy image paths
    noisy_images = {}
    for noise_dir in noisy_dirs:
        for celeb_folder in os.listdir(noise_dir):
            noisy_celeb_path = os.path.join(noise_dir, celeb_folder)
            if not os.path.isdir(noisy_celeb_path):
                print(f"Skipping non-directory: {noisy_celeb_path}")
                continue
            
            noisy_paths = sorted(
                glob.glob(os.path.join(noisy_celeb_path, "*.jpg")) +
                glob.glob(os.path.join(noisy_celeb_path, "*.png")) +
                glob.glob(os.path.join(noisy_celeb_path, "*.jpeg"))
            )
            
            for noisy_path in noisy_paths:
                noisy_name = os.path.splitext(os.path.basename(noisy_path))[0]
                noisy_images.setdefault(noise_dir, {}).setdefault(celeb_folder, {})[noisy_name] = noisy_path
    
    # Create image pairs
    image_pairs = []
    unmatched_clean = []
    unmatched_noisy = []
    
    for celeb_folder in clean_images:
        for clean_name, clean_path in clean_images[celeb_folder].items():
            found_pair = False
            for noise_dir in noisy_dirs:
                if (noise_dir in noisy_images and
                    celeb_folder in noisy_images[noise_dir] and
                    clean_name in noisy_images[noise_dir][celeb_folder]):
                    noisy_path = noisy_images[noise_dir][celeb_folder][clean_name]
                    image_pairs.append((noisy_path, clean_path))
                    found_pair = True
            if not found_pair:
                unmatched_clean.append((celeb_folder, clean_name, clean_path))
    
    # Log unmatched noisy images
    for noise_dir in noisy_images:
        for celeb_folder in noisy_images[noise_dir]:
            for noisy_name, noisy_path in noisy_images[noise_dir][celeb_folder].items():
                if (celeb_folder not in clean_images or
                    noisy_name not in clean_images[celeb_folder]):
                    unmatched_noisy.append((noise_dir, celeb_folder, noisy_name, noisy_path))
    
    if not image_pairs:
        raise ValueError("No valid image pairs found. Check directory paths and file naming.")
    
    print(f"Total pairs: {len(image_pairs)}")

    if unmatched_clean:
        print(f"Unmatched clean images: {len(unmatched_clean)}")
        print("Sample unmatched clean:", unmatched_clean[:5])
    if unmatched_noisy:
        print(f"Unmatched noisy images: {len(unmatched_noisy)}")
        print("Sample unmatched noisy:", unmatched_noisy[:5])
    
    def image_generator():
        for noisy_path, clean_path in image_pairs:
            noisy, clean = load_and_preprocess(noisy_path, clean_path)
            yield noisy, clean

    dataset = tf.data.Dataset.from_generator(
        image_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((256, 256, 3), (256, 256, 3))
    )
    
    # Apply shuffle but not batching
    dataset = dataset.shuffle(buffer_size=5000)
    
    # Save the unbatched dataset to disk
    print(f"Saving dataset to {cache_dir}")
    
    dataset.save(cache_dir)
    
    return dataset

if __name__ == '__main__':
    noisy_dirs = [
        "D:/dataset/noisy/gaussian",
        "D:/dataset/noisy/salt_pepper",
        "D:/dataset/noisy/uniform",
        "D:/dataset/noisy/poisson",
        "D:/dataset/noisy/speckle"
    ]
    clean_dir = "D:/dataset/clean"
    dataset = prepare_data(noisy_dirs, clean_dir)
    
    # Test the dataset (unbatched)
    for noisy, clean in dataset.take(1):
        print("Noisy shape:", noisy.shape)
        print("Clean shape:", clean.shape)