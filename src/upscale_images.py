import os
import glob
import cv2
from cv2 import dnn_superres
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT_DIR, 'data', 'images_224_base')
OUTPUT_DIR_EDSR = os.path.join(ROOT_DIR, 'data', 'images_896_edsr')
OUTPUT_DIR_BICUBIC = os.path.join(ROOT_DIR, 'data', 'images_896_bicubic')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'EDSR_x4.pb')

# Target Folders & Limits (As discussed: 200 images for 5000 sizes)
TARGET_FOLDERS = ['Img_5000']
LIMIT_PER_CLASS = 200

def process_image(args):
    input_path, out_edsr, out_bicubic = args
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error reading {input_path}"

        # 1. Bicubic
        h, w = img.shape[:2]
        img_bicubic = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(out_bicubic, img_bicubic)

        # 2. EDSR (CUDA Accelerated)
        img_edsr = sr.upsample(img)
        cv2.imwrite(out_edsr, img_edsr)
        return None
    except Exception as e:
        return f"Error: {str(e)}"

sr = None
def init_worker(model_path):
    global sr
    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", 4)
        
        # --- ENABLE CUDA ACCELERATION ---
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except Exception as e:
        print(f"Worker init failed (Is OpenCV CUDA-enabled?): {e}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    tasks = []
    for folder_name in TARGET_FOLDERS:
        src_folder = os.path.join(INPUT_DIR, folder_name)
        if not os.path.exists(src_folder): continue
        
        sat_folders = glob.glob(os.path.join(src_folder, '*'))
        for sat_folder in sat_folders:
            sat_name = os.path.basename(sat_folder)
            dst_edsr = os.path.join(OUTPUT_DIR_EDSR, folder_name, sat_name)
            dst_bicubic = os.path.join(OUTPUT_DIR_BICUBIC, folder_name, sat_name)
            os.makedirs(dst_edsr, exist_ok=True)
            os.makedirs(dst_bicubic, exist_ok=True)
            
            # Sort & Limit 200
            images = sorted(glob.glob(os.path.join(sat_folder, '*.png')), 
                           key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            images = images[:LIMIT_PER_CLASS]
            
            for img_path in images:
                img_name = os.path.basename(img_path)
                out_e = os.path.join(dst_edsr, img_name)
                out_b = os.path.join(dst_bicubic, img_name)
                if not (os.path.exists(out_e) and os.path.exists(out_b)):
                    tasks.append((img_path, out_e, out_b))

    print(f"Tasks: {len(tasks)} images (Mode: CUDA Accelerated)")
    if len(tasks) == 0: return

    # For GPU processing, too many workers can cause OOM (Out of Memory).
    # 2-4 workers are usually enough for one GPU.
    num_workers = 2 
    
    try:
        set_start_method('spawn', force=True)
    except RuntimeError: pass

    with Pool(processes=num_workers, initializer=init_worker, initargs=(MODEL_PATH,)) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))
        
    print("Done.")

if __name__ == '__main__':
    main()