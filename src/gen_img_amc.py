import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data/xlsx')
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, 'data', 'images_256_amc') # AMC paper usually uses slightly larger or specific dims

SAMPLE_SIZES = [500, 1000, 2000, 5000] 
IMAGE_SIZE_H = 256 # Height (Radius)
IMAGE_SIZE_W = 256 # Width (Angle)

# Polar Plot Ranges
# Radius: Normalized 0 to 1 (or slightly more)
# Angle: -pi to pi
RANGE_R_MIN, RANGE_R_MAX = 0.0, 1.2 # slightly > 1.0 to catch outliers
RANGE_THETA_MIN, RANGE_THETA_MAX = -np.pi, np.pi

def process_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        sat_name = os.path.splitext(file_name)[0]
        
        # Read Data
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
        
        complex_data = []
        for val in df.values.flatten():
            if pd.isna(val): continue
            if isinstance(val, str):
                try:
                    val = val.replace('i', 'j')
                    c = complex(val)
                    complex_data.append(c)
                except ValueError: continue
            elif isinstance(val, (int, float, complex)):
                complex_data.append(val)
        
        complex_data = np.array(complex_data, dtype=np.complex128)
        
        if len(complex_data) == 0:
            return f"Skipped {sat_name}: No data"

        # 1. Normalization (Same as Base, essential for comparison)
        real_part = complex_data.real
        imag_part = complex_data.imag
        
        min_i, max_i = real_part.min(), real_part.max()
        min_q, max_q = imag_part.min(), imag_part.max()
        
        if max_i == min_i: max_i += 1e-9
        if max_q == min_q: max_q += 1e-9

        # Normalize to [-1, 1] first
        i_norm = 2 * ((real_part - min_i) / (max_i - min_i)) - 1
        q_norm = 2 * ((imag_part - min_q) / (max_q - min_q)) - 1
        
        # 2. Polar Transformation
        # r = sqrt(i^2 + q^2)
        # theta = arctan2(q, i)
        r_vals = np.sqrt(i_norm**2 + q_norm**2)
        theta_vals = np.arctan2(q_norm, i_norm)
        
        # Process for each sample size
        for n_samples in SAMPLE_SIZES:
            output_dir = os.path.join(OUTPUT_BASE_DIR, f'Img_{n_samples}', sat_name)
            os.makedirs(output_dir, exist_ok=True)
            
            num_images = len(complex_data) // n_samples
            
            for i in range(num_images):
                start_idx = i * n_samples
                end_idx = (i + 1) * n_samples
                
                chunk_r = r_vals[start_idx:end_idx]
                chunk_theta = theta_vals[start_idx:end_idx]
                
                # 2D Histogram in Polar Coordinates
                # Y-axis: Radius (r), X-axis: Angle (theta)
                # Why? Because in CNN images, vertical features (lines) are often easier to process,
                # and phase noise (rotation) becomes a vertical shift if Theta is Y, or horizontal if Theta is X.
                # AMC paper: "phase imperfections appear as time shifts... phase acts as shift factor on phase component"
                # Let's map Theta to X-axis and Radius to Y-axis.
                
                H, _, _ = np.histogram2d(chunk_theta, chunk_r, 
                                         bins=[IMAGE_SIZE_W, IMAGE_SIZE_H], 
                                         range=[[RANGE_THETA_MIN, RANGE_THETA_MAX], [RANGE_R_MIN, RANGE_R_MAX]])
                
                # Normalize [0, 1]
                h_min, h_max = H.min(), H.max()
                if h_max > h_min:
                    H_norm = (H - h_min) / (h_max - h_min)
                else:
                    H_norm = H

                # Transpose for Image (Row=Y=Radius, Col=X=Theta)
                # Note: histogram2d returns H[x, y]. 
                # If we want Theta on X (Cols) and R on Y (Rows), we need H.T
                # Correct orientation: Y-axis (Rows) = Radius, X-axis (Cols) = Theta
                img_data = H_norm.T 
                
                # Flip Y (Radius) so 0 is at bottom? Standard images 0,0 is top-left.
                # Usually R=0 at bottom is intuitive for plots, but for CNN features consistency matters most.
                # Let's flip so R=0 is bottom (high index) to match Cartesian plot intuition roughly.
                img_data = np.flipud(img_data)
                
                # Convert to uint8 [0, 255]
                img_uint8 = (img_data * 255).astype(np.uint8)
                
                # Convert to RGB
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
