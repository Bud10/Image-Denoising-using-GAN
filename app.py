from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Enhancement API")

# Path to the trained generator model
MODEL_PATH = "C:/Users/Acer/Downloads/Model_info/checkpoints_100ep/generator_epoch_100.keras"

# Load the model at startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise ValueError(f"Failed to load model: {e}")

def extract_patches(image: np.ndarray, patch_size: int = 256, stride: int = 128) -> tuple[list, list, list]:
    """Extract overlapping patches from the image."""
    try:
        height, width = image.shape[:2]
        patches = []
        h_positions = []
        w_positions = []
        
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch = image[i:i+patch_size, j:j+patch_size, :]
                patches.append(patch)
                h_positions.append(i)
                w_positions.append(j)
        logger.debug(f"Extracted {len(patches)} patches from {width}x{height} image")
        return patches, h_positions, w_positions
    except Exception as e:
        logger.error(f"Error in extract_patches: {e}")
        raise

def blend_patches(enhanced_patches: list, h_positions: list, w_positions: list, height: int, width: int, patch_size: int = 256) -> np.ndarray:
    """Blend patches back into a full-size image with feathering."""
    try:
        output = np.zeros((height, width, 3), dtype=np.float32)
        weight_sum = np.zeros((height, width, 1), dtype=np.float32)
        
        # Create a blending weight mask (linear fall-off)
        weight_mask = np.ones((patch_size, patch_size, 1), dtype=np.float32)
        ramp = np.linspace(0, 1, patch_size // 4)
        for k in range(patch_size // 4):
            weight_mask[k, :] *= ramp[k]
            weight_mask[-(k+1), :] *= ramp[k]
            weight_mask[:, k] *= ramp[k]
            weight_mask[:, -(k+1)] *= ramp[k]
        
        for patch, i, j in zip(enhanced_patches, h_positions, w_positions):
            output[i:i+patch_size, j:j+patch_size, :] += patch * weight_mask
            weight_sum[i:i+patch_size, j:j+patch_size, :] += weight_mask
        
        # Normalize by weight sum
        output /= np.where(weight_sum > 0, weight_sum, 1)
        logger.debug(f"Blended {len(enhanced_patches)} patches into {width}x{height} image")
        return output
    except Exception as e:
        logger.error(f"Error in blend_patches: {e}")
        raise

def preprocess_patch(patch: np.ndarray) -> np.ndarray:
    """Preprocess a single patch for the GAN model."""
    try:
        img_array = patch.astype(np.float32)
        img_array = (img_array - 127.5) / 127.5
        img_array = np.expand_dims(img_array, axis=0)
        logger.debug(f"Preprocessed patch shape: {img_array.shape}, dtype: {img_array.dtype}")
        return img_array
    except Exception as e:
        logger.error(f"Error in preprocess_patch: {e}")
        raise

def postprocess_patch(img_array: np.ndarray) -> np.ndarray:
    """Convert model output to image array."""
    try:
        img_array = img_array[0]
        img_array = (img_array * 127.5 + 127.5).clip(0, 255)
        logger.debug(f"Postprocessed patch shape: {img_array.shape}, dtype: {img_array.dtype}")
        return img_array.astype(np.float32)
    except Exception as e:
        logger.error(f"Error in postprocess_patch: {e}")
        raise

@app.post("/enhance", response_class=StreamingResponse)
async def enhance_image(file: UploadFile = File(...)):
    """Enhance an uploaded image at full size using patch-based processing."""
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if file.size > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        # Read and open image
        contents = await file.read()
        logger.debug(f"File size: {len(contents)} bytes")
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape[:2]
        logger.info(f"Processing image of size: {width}x{height}")
        
        if height < 256 or width < 256:
            raise HTTPException(status_code=400, detail="Image too small; minimum size is 256x256")
        
        # Extract patches
        patches, h_positions, w_positions = extract_patches(img_array, patch_size=256, stride=128)
        
        # Enhance patches
        enhanced_patches = []
        for patch in patches:
            patch_input = preprocess_patch(patch)
            enhanced_patch = model.predict(patch_input, verbose=0)
            enhanced_patches.append(postprocess_patch(enhanced_patch))
        
        # Blend patches
        enhanced_image = blend_patches(enhanced_patches, h_positions, w_positions, height, width)
        
        # Convert to PIL Image
        enhanced_image = enhanced_image.astype(np.uint8)
        enhanced_pil = Image.fromarray(enhanced_image)
        
        # Save to in-memory buffer
        img_byte_arr = io.BytesIO()
        enhanced_pil.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        logger.info("Prepared in-memory PNG buffer for response")
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=enhanced_image.png"}
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Image Enhancement API is running"}