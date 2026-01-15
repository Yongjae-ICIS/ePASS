import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src/
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data/xlsx')
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, 'data', 'images_224_base')

SAMPLE_SIZES = [500, 1000, 2000, 5000] 
IMAGE_SIZE = 224
RANGE_MIN, RANGE_MAX = -1.5, 1.5

def process_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        sat_name = os.path.splitext(file_name)[0]
        
        # Read Excel file
        # Using openpyxl engine for .xlsx, header=None because matlab code said ReadVariableNames=false
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
        
        # Flatten and convert to complex
        # The MATLAB code iterates through cells and converts strings/numbers.
        # We'll assume the dataframe might contain mixed types.
        complex_data = []
        for val in df.values.flatten():
            if pd.isna(val):
                continue
            if isinstance(val, str):
                try:
                    # Remove 'i' or 'j' ambiguity if necessary, but complex() usually handles 'j'
                    # Matlab's str2num handles 'i'. Python handles 'j'.
                    val = val.replace('i', 'j') 
                    c = complex(val)
                    complex_data.append(c)
                except ValueError:
                    continue
            elif isinstance(val, (int, float, complex)):
                complex_data.append(val)
        
        complex_data = np.array(complex_data, dtype=np.complex128)
        
        if len(complex_data) == 0:
            return f"Skipped {sat_name}: No data"

        # Per-Satellite Normalization [-1, 1]
        real_part = complex_data.real
        imag_part = complex_data.imag
        
        min_i, max_i = real_part.min(), real_part.max()
        min_q, max_q = imag_part.min(), imag_part.max()
        
        # Avoid division by zero
        if max_i == min_i: max_i += 1e-9
        if max_q == min_q: max_q += 1e-9

        real_norm = 2 * ((real_part - min_i) / (max_i - min_i)) - 1
        imag_norm = 2 * ((imag_part - min_q) / (max_q - min_q)) - 1
        
        # Process for each sample size
        for n_samples in SAMPLE_SIZES:
            output_dir = os.path.join(OUTPUT_BASE_DIR, f'Img_{n_samples}', sat_name)
            os.makedirs(output_dir, exist_ok=True)
            
            num_images = len(complex_data) // n_samples
            
            for i in range(num_images):
                start_idx = i * n_samples
                end_idx = (i + 1) * n_samples
                
                chunk_i = real_norm[start_idx:end_idx]
                chunk_q = imag_norm[start_idx:end_idx]
                
                # 2D Histogram
                # Range is [-1.5, 1.5] as per matlab code
                H, _, _ = np.histogram2d(chunk_i, chunk_q, bins=IMAGE_SIZE, 
                                         range=[[RANGE_MIN, RANGE_MAX], [RANGE_MIN, RANGE_MAX]])
                
                # Normalize [0, 1] (Min-Max scaling on the histogram itself)
                h_min, h_max = H.min(), H.max()
                if h_max > h_min:
                    H_norm = (H - h_min) / (h_max - h_min)
                else:
                    H_norm = H # or zeros
                
                # Resize is not needed if bins=IMAGE_SIZE, but MATLAB code did imresize? 
                # Actually histogram2d already gives the grid. 
                # Note: numpy histogram origin is bottom-left, typically. Image is top-left.
                # We should rotate/flip to match standard image coordinates if needed, 
                # but for ML it might not matter as long as it's consistent.
                # Let's keep it simple: Transpose to make x=I, y=Q match image (row=y, col=x)
                H_norm = H_norm.T 
                
                # Convert to uint8 [0, 255]
                img_uint8 = (H_norm * 255).astype(np.uint8)
                
                # Convert to RGB (3 channels)
                # In MATLAB: repmat(..., [1, 1, 3])
                img_rgb = np.stack((img_uint8,)*3, axis=-1)
                
                # Save
                save_path = os.path.join(output_dir, f'{sat_name}_image_{i+1}.png')
                Image.fromarray(img_rgb).save(save_path)

        return f"Processed {sat_name}: {num_images} images (size {n_samples})"

    except Exception as e:
        return f"Error {sat_name}: {str(e)}"

def main():
    files = glob.glob(os.path.join(RAW_DATA_DIR, '*.xlsx'))
    print(f"Found {len(files)} files in {RAW_DATA_DIR}")
    
    if not files:
        print("No files found. Please check the directory.")
        return

    # Parallel processing
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))
        
    for res in results:
        print(res)

if __name__ == '__main__':
    main()
