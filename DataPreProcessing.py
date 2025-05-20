import tensorflow as tf
import cv2
import numpy as np
import os
import glob
import shutil

#Loads images, resizes to (256,256) then normalizes to [-1,1] range
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

 #Creates matching pairs of images and saves in dataset/cache
def prepare_data(noisy_dirs, clean_dir, cache_dir="D:/dataset/cache"):
    # Clear existing cache directory to ensure overwrite
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    clean_paths = sorted(
        glob.glob(os.path.join(clean_dir, "*.png")) +
        glob.glob(os.path.join(clean_dir, "*.jpg")) +
        glob.glob(os.path.join(clean_dir, "*.jpeg"))
    )
    noisy_paths = []
    for noise_dir in noisy_dirs:
        noisy_paths.extend(sorted(
            glob.glob(os.path.join(noise_dir, "*.png")) +
            glob.glob(os.path.join(noise_dir, "*.jpg")) +
            glob.glob(os.path.join(noise_dir, "*.jpeg"))
        ))

    image_pairs = []
    for clean_path in clean_paths:
        clean_name = os.path.splitext(os.path.basename(clean_path))[0]
        for noise_dir in noisy_dirs:
            for ext in [".png", ".jpg", ".jpeg"]:
                noisy_path = os.path.join(noise_dir, clean_name + ext)
                if os.path.exists(noisy_path):
                    image_pairs.append((noisy_path, clean_path))
    
    if not image_pairs:
        raise ValueError("No valid image pairs found. Check directory paths and file naming.")
    
    print(f"Total pairs: {len(image_pairs)}")
    print("Sample pairs:", image_pairs[:5])

    #Images are generated using load_and_preprocess function
    def image_generator():
        for noisy_path, clean_path in image_pairs:
            noisy, clean = load_and_preprocess(noisy_path, clean_path)
            yield noisy, clean
    #Dataset of tensor is created using list of paired images
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
