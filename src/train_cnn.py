import os
import argparse
import time
import copy
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_ROOT = os.path.join(ROOT_DIR, 'data', 'images_224_base')
AMC_DATA_ROOT = os.path.join(ROOT_DIR, 'data', 'images_256_amc')
EDSR_DATA_ROOT = os.path.join(ROOT_DIR, 'data', 'images_896_edsr')
BICUBIC_DATA_ROOT = os.path.join(ROOT_DIR, 'data', 'images_896_bicubic')

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class SatelliteDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        try:
            image = Image.open(self.file_paths[idx]).convert('RGB')
        except: image = Image.new('RGB', (224, 224))
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

def get_data_loaders(data_dir, batch_size=32, input_size=224, limit_per_class=200, split_strategy='random'):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []
    
    print(f"Scanning data ({split_strategy.upper()}, Limit={limit_per_class}) from: {data_dir}")
    
    for cls_name in tqdm(classes, desc="Classes"):
        cls_dir = os.path.join(data_dir, cls_name)
        files = sorted(glob.glob(os.path.join(cls_dir, '*.png')),
                       key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        files = files[:limit_per_class]
        if len(files) < 10: continue
        
        cls_idx = class_to_idx[cls_name]
        labels = [cls_idx] * len(files)
        
        if split_strategy == 'sequential':
            n_train = int(len(files) * 0.6); n_val = int(len(files) * 0.2)
            train_files.extend(files[:n_train]); train_labels.extend(labels[:n_train])
            val_files.extend(files[n_train:n_train+n_val]); val_labels.extend(labels[n_train:n_train+n_val])
            test_files.extend(files[n_train+n_val:]); test_labels.extend(labels[n_train+n_val:])
        else:
            f_train, f_temp, l_train, l_temp = train_test_split(files, labels, test_size=0.4, random_state=42)
            f_val, f_test, l_val, l_test = train_test_split(f_temp, l_temp, test_size=0.5, random_state=42)
            train_files.extend(f_train); train_labels.extend(l_train)
            val_files.extend(f_val); val_labels.extend(l_val)
            test_files.extend(f_test); test_labels.extend(l_test)

    data_transforms = {
        x: transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) for x in ['train', 'val', 'test']
    }
    
    image_datasets = {
        'train': SatelliteDataset(train_files, train_labels, data_transforms['train']),
        'val': SatelliteDataset(val_files, val_labels, data_transforms['val']),
        'test': SatelliteDataset(test_files, test_labels, data_transforms['test'])
    }
    
    # Use 8 workers if CUDA, 0 if MPS/CPU
    is_cuda = torch.cuda.is_available()
    workers = 8 if is_cuda else 0
    
    return {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=workers, pin_memory=is_cuda) for x in ['train', 'val', 'test']}, len(classes)

def train_one_config(mode, sample_size, epochs, batch_size, device, limit, split_strategy):
    mode_map = {
        'base': (BASE_DATA_ROOT, 224),
        'amc': (AMC_DATA_ROOT, 256),
        'edsr': (EDSR_DATA_ROOT, 224), # Resize to 224
        'bicubic': (BICUBIC_DATA_ROOT, 224)
    }
    root, input_size = mode_map[mode]
    data_dir = os.path.join(root, f'Img_{sample_size}')
    if not os.path.exists(data_dir):
        print(f"Skipping {mode.upper()} {sample_size}: Directory not found.")
        return 0.0

    print(f"\n>>> RUNNING: {mode.upper()} | Samples: {sample_size} | Split: {split_strategy.upper()}")
    dataloaders, num_classes = get_data_loaders(data_dir, batch_size, input_size, limit, split_strategy)
    
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(); optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_model_wts = copy.deepcopy(model.state_dict()); best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs); loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train': loss.backward(); optimizer.step()
                running_corrects += torch.sum(preds == labels.data)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            print(f'{phase} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc; best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts); model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0); correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    print(f'Final Test Acc: {test_acc:.2f}%')
    with open(os.path.join(RESULTS_DIR, f'result_{mode}_{sample_size}_L{limit}_{split_strategy}.txt'), 'w') as f:
        f.write(f"Acc: {test_acc:.2f}%\n")
    return test_acc

def main():
    parser = argparse.ArgumentParser()
    # Default modes: both bicubic and edsr
    parser.add_argument('--mode', type=str, nargs='+', default=['bicubic', 'edsr'], choices=['base', 'amc', 'edsr', 'bicubic'])
    # Default samples: 500, 1000
    parser.add_argument('--samples', type=int, nargs='+', default=[500, 1000])
    # Default epochs: 1
    parser.add_argument('--epochs', type=int, default=1)
    # Default limit: 200
    parser.add_argument('--limit', type=int, default=200)
    # Default split: random
    parser.add_argument('--split_strategy', type=str, default='random', choices=['sequential', 'random'])
    # Batch size reverted to 32 for better performance
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    all_results = {}
    for m in args.mode:
        for s in args.samples:
            acc = train_one_config(m, s, args.epochs, args.batch_size, device, args.limit, args.split_strategy)
            all_results[f"{m}_{s}"] = acc

    print("\n" + "="*30)
    print("FINAL SUMMARY")
    print("="*30)
    for k, acc in all_results.items():
        print(f"{k:15}: {acc:.2f}%")

if __name__ == '__main__':
    main()
