import os
import argparse
import copy
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define Data Paths
DATA_PATHS = {
    'base': os.path.join(ROOT_DIR, 'data', 'images_224_base'),
    'edsr': os.path.join(ROOT_DIR, 'data', 'images_896_edsr'),
    'bicubic': os.path.join(ROOT_DIR, 'data', 'images_896_bicubic')
}
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Dataset ---
class SatelliteDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        try: image = Image.open(self.file_paths[idx]).convert('RGB')
        except: image = Image.new('RGB', (224, 224))
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

def get_dataloaders(data_dir, batch_size, input_size, limit):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    train_f, train_l, val_f, val_l, test_f, test_l = [], [], [], [], [], []
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        # Sort for deterministic limiting
        files = sorted(glob.glob(os.path.join(cls_dir, '*.png')),
                       key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        files = files[:limit]
        if len(files) < 10: continue
        
        labels = [class_to_idx[cls]] * len(files)
        
        # Random Split (Same logic as train_model.py)
        ft, f_tmp, lt, l_tmp = train_test_split(files, labels, test_size=0.4, random_state=42)
        fv, f_test, lv, l_test = train_test_split(f_tmp, l_tmp, test_size=0.5, random_state=42)
        
        train_f.extend(ft); train_l.extend(lt)
        val_f.extend(fv); val_l.extend(lv)
        test_f.extend(f_test); test_l.extend(l_test)
        
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(), # (C, H, W) -> range [0, 1]
        # No normalization for RNN usually, or simple 0.5 mean
    ])
    
    dsets = {
        'train': SatelliteDataset(train_f, train_l, transform),
        'val': SatelliteDataset(val_f, val_l, transform),
        'test': SatelliteDataset(test_f, test_l, transform)
    }
    
    return {x: DataLoader(dsets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=0) for x in ['train', 'val', 'test']}, len(classes)

# --- Models ---
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, rnn_type='lstm'):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, C, H, W)
        # We treat image as sequence of rows.
        # Reshape to (Batch, H, W*C) -> Sequence Length = H, Input Size = W*C
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        x = x.view(b, h, w*c) # (B, Seq_Len, Input_Size)
        
        # RNN Forward
        # out: (Batch, Seq_Len, Hidden)
        out, _ = self.rnn(x)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train(model_type, data_mode, sample_size, epochs, limit, device):
    data_dir = os.path.join(DATA_PATHS[data_mode], f'Img_{sample_size}')
    if not os.path.exists(data_dir):
        print(f"Skipping {data_dir} (Not Found)")
        return 0.0
        
    print(f"\n>>> RNN Training: {model_type.upper()} | Data: {data_mode} | Size: {sample_size}")
    
    # Input size 224 (image height/width). 
    # RNN input dim = 224 * 3 (RGB) = 672
    IMG_SIZE = 224
    INPUT_DIM = IMG_SIZE * 3 
    HIDDEN_DIM = 128
    LAYERS = 2
    BATCH_SIZE = 32
    
    loaders, num_classes = get_dataloaders(data_dir, BATCH_SIZE, IMG_SIZE, limit)
    
    model = RNNClassifier(INPUT_DIM, HIDDEN_DIM, LAYERS, num_classes, model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            correct = 0
            total = 0
            
            for imgs, labels in tqdm(loaders[phase], desc=phase):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(imgs)
                    loss = criterion(out, labels)
                    _, preds = torch.max(out, 1)
                    if phase == 'train':
                        loss.backward(); optimizer.step()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            acc = correct / total
            print(f"{phase} Acc: {acc:.4f}")
            if phase == 'val' and acc > best_acc:
                best_acc = acc; best_wts = copy.deepcopy(model.state_dict())
                
    # Test
    model.load_state_dict(best_wts)
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for imgs, labels in loaders['test']:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    test_acc = 100 * correct / total
    print(f"Final Test Acc: {test_acc:.2f}%")
    
    with open(os.path.join(RESULTS_DIR, f'result_rnn_{model_type}_{data_mode}_{sample_size}.txt'), 'w') as f:
        f.write(f"{test_acc:.2f}")
    return test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, nargs='+', default=['lstm', 'gru'])
    # Default mode: only EDSR as requested
    parser.add_argument('--mode', type=str, nargs='+', default=['edsr'], choices=['base', 'amc', 'edsr', 'bicubic'])
    parser.add_argument('--samples', type=int, nargs='+', default=[500, 1000])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    results = {}
    for m_data in args.mode:
        for m_type in args.model:
            for s in args.samples:
                acc = train(m_type, m_data, s, args.epochs, args.limit, device)
                results[f"{m_data}_{m_type}_{s}"] = acc
            
    print("\nSummary:")
    for k, v in results.items():
        print(f"{k}: {v:.2f}%")

if __name__ == '__main__':
    main()