import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(256, 256, 3)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(3, (3, 3), padding='same', activation='tanh'))
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(256, 256, 3)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Combined GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
perceptual_loss = tf.keras.losses.MeanAbsoluteError()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output, generated_image, target_image):
    gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    perc_loss = perceptual_loss(generated_image, target_image)
    return gan_loss + 100 * perc_loss

# Training step
@tf.function
def train_step(noisy_images, clean_images, generator, discriminator, gan, g_optimizer, d_optimizer):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        generated_images = generator(noisy_images, training=True)
        
        real_output = discriminator(clean_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        g_loss = generator_loss(fake_output, generated_images, clean_images)
        d_loss = discriminator_loss(real_output, fake_output)
    
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    
    return g_loss, d_loss

# Test model function
def test_model(generator, dataset, split_name, num_samples=3, epoch=0, output_dir="D:/dataset/test_output"):
    """Test the model by generating denoised images and computing metrics."""
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
    
    samples = list(dataset.take(num_samples))
    if not samples:
        raise ValueError(f"{split_name} dataset is empty. Cannot perform testing.")
    
    psnr_values = []
    ssim_values = []
    
    for i, (noisy, clean) in enumerate(samples):
        generated = generator(noisy[None, ...], training=False)[0]
        
        noisy_np = (noisy.numpy() + 1) / 2
        clean_np = (clean.numpy() + 1) / 2
        generated_np = (generated.numpy() + 1) / 2
        
        psnr_value = psnr(clean_np, generated_np, data_range=1.0)
        ssim_value = ssim(clean_np, generated_np, data_range=1.0, multichannel=True)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(noisy_np)
        plt.title(f"Noisy Image {i+1} ({split_name})")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(clean_np)
        plt.title(f"Clean Image {i+1} ({split_name})")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(generated_np)
        plt.title(f"Denoised Image {i+1} ({split_name})\nPSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, split_name, f"test_epoch_{epoch}_sample_{i+1}.png"))
        plt.close()
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Test on {split_name} at epoch {epoch}: Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.3f}")
    return avg_psnr, avg_ssim

# Split dataset function
def split_dataset(dataset, train_split=0.8, val_split=0.1):
    """Split dataset into train, validation, and test sets."""
    # Convert dataset to list for splitting
    data_list = list(dataset)
    if not data_list:
        raise ValueError("Dataset is empty. Cannot split.")
    
    print(f"Total samples: {len(data_list)}")
    
    # Split into train, validation, and test
    train_data, temp_data = train_test_split(data_list, train_size=train_split, random_state=None)  # No seed for randomness
    val_data, test_data = train_test_split(temp_data, train_size=val_split/(1-train_split), random_state=None)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Convert back to datasets
    train_dataset = tf.data.Dataset.from_generator(
        lambda: iter(train_data),
        output_types=(tf.float32, tf.float32),
        output_shapes=((256, 256, 3), (256, 256, 3))
    )
    val_dataset = tf.data.Dataset.from_generator(
        lambda: iter(val_data),
        output_types=(tf.float32, tf.float32),
        output_shapes=((256, 256, 3), (256, 256, 3))
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: iter(test_data),
        output_types=(tf.float32, tf.float32),
        output_shapes=((256, 256, 3), (256, 256, 3))
    )
    
    return train_dataset, val_dataset, test_dataset

# Training loop
def train_gan(train_dataset, val_dataset, epochs, generator, discriminator, gan, batch_size=32, checkpoint_dir="D:/dataset/checkpoints"):
    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_psnr = 0.0
    
    for epoch in range(epochs):
        for batch in train_dataset:
            g_loss, d_loss = train_step(batch[0], batch[1], generator, discriminator, gan, g_optimizer, d_optimizer)
        
        print(f'Epoch {epoch+1}, Gen Loss: {g_loss:.4f}, Disc Loss: {d_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            val_psnr, val_ssim = test_model(generator, val_dataset, "val", num_samples=3, epoch=epoch+1)
            
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                generator.save(os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.h5"))
                print(f"Saved checkpoint with PSNR {val_psnr:.2f} at epoch {epoch+1}")

# Main execution
if __name__ == '__main__':
    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    # Compile models
    generator.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    # Load cached dataset
    cache_dir = "D:/dataset/cache"
    element_spec = (
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.load(cache_dir, element_spec=element_spec)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_split=0.8, val_split=0.1)
    
    # Apply batching and prefetching
    batch_size = 32  # Can be changed to any batch size
    train_dataset = train_dataset.shuffle(buffer_size=5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Train the model
    train_gan(train_dataset, val_dataset, epochs=100, generator=generator, discriminator=discriminator, gan=gan, batch_size=batch_size)
    
    # Final evaluation on test dataset
    test_model(generator, test_dataset, "test", num_samples=3, epoch=100)