# Image-Denoising-using-GAN

## DataPreProcessing File
Here, the Images are taken from clean directory and noisy directory which contains 5 noises Gaussian, salt_and_pepper, poisson, speckle and uniform.
Then, images are resized to (256,256,3) dimensions.
Each clean image has 5 noisy images. So, a list is made pairing clean and noisy image.
This list is stored as tensor dataset and stored in cache directory.

## Training File
Contains generator and discriminator code.
Loads the dataset from cache dir.
splits the data into 80% training, 10% testing, 10% validation.
