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

##Result
The result presents a comparative analysis of Generative Adversarial 
Network (GAN)-based image denoising models, including CGAN, SRGAN, ESRGAN, and a 
proposed Denoising GAN framework. Performance evaluation is 
conducted using PSNR, SSIM, MS-SSIM, and LPIPS metrics to assess reconstruction quality 
and perceptual similarity. 
Experimental results demonstrate that the proposed Denoising 
GAN outperforms baseline models, achieving approximately 32 PSNR, 0.90 SSIM, and 0.98 
MS-SSIM while maintaining the lowest LPIPS score of 0.05, indicating superior structural preservation and perceptual quality. 
Training stability analysis further shows balanced 
adversarial convergence. The results highlight the effectiveness of the proposed model for 
high-resolution image restoration in applications such as media publishing, facial 
recognition, and digital archival enhancement.
<img width="1103" height="516" alt="img2" src="https://github.com/user-attachments/assets/c5c8c39a-f8e3-49b1-adf1-7dfca4d0b85d" />
