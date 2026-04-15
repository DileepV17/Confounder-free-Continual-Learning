import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Function
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import generate_data
from synthetic_dataset import SyntheticDataset
import random
import os

def set_seed(seed=42):
    """Locks down all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Forces cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DiversityEnsembleDANN(nn.Module):
    def __init__(self, num_subnets=3):
        super().__init__()
        self.num_subnets = num_subnets
        
        # 1. Shared Feature Extractor (Backbone)
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # 2. Independent Subnetworks (K=3)
        # We use a ModuleList to keep them separate
        self.subnets = nn.ModuleList([
            nn.Sequential(nn.Linear(32 * 5 * 5, 84), nn.ReLU()) 
            for _ in range(num_subnets)
        ])
        
        # 3. Task Classifier (Takes the mean of the subnetworks)
        self.classifier = nn.Linear(84, 2)
        
        # 4. Domain Classifier for DANN (Make it stronger with a hidden layer!)
        self.domain_classifier = nn.Sequential(
            nn.Linear(84, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, x, alpha=1.0):
        # Base features
        base_features = self.shared_encoder(x)
        
        # Pass through all K subnetworks
        subnet_outputs = [subnet(base_features) for subnet in self.subnets]
        
        # Stack outputs to shape (batch_size, num_subnets, feature_dim)
        stacked_outputs = torch.stack(subnet_outputs, dim=1)
        
        # Calculate the mean embedding across subnetworks
        mean_embedding = stacked_outputs.mean(dim=1)
        
        # Task prediction
        class_preds = self.classifier(mean_embedding)
        
        # Domain prediction (Adversarial GRL applied here)
        rev_features = GradientReversal.apply(mean_embedding, alpha)
        conf_preds = self.domain_classifier(rev_features)
        
        return class_preds, conf_preds, stacked_outputs




# ==========================================
# 2. DANN ARCHITECTURE (BR-NET EQUIVALENT)
# ==========================================
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DANN_Continual(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D CNN Backbone as described in the paper's appendix
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 5 * 5, 84), nn.ReLU(), nn.Linear(84, 2)
        )
        # Confounder Predictor (Regression for continuous confounder)
        self.domain_classifier = nn.Sequential(
            nn.Linear(32 * 5 * 5, 84), nn.ReLU(), nn.Linear(84, 1)
        )

    def forward(self, x, alpha=1.0):
        features = self.extractor(x)
        class_preds = self.classifier(features)
        
        rev_features = GradientReversal.apply(features, alpha)
        conf_preds = self.domain_classifier(rev_features)
        
        return class_preds, conf_preds
    

class SquaredCorrelationLoss(nn.Module):
    def __init__(self):
        super(SquaredCorrelationLoss, self).__init__()

    def forward(self, pred, target):
        # Ensure target is the same shape as pred
        target = target.view_as(pred)
        
        # Subtract the mean
        pred_mean = pred - torch.mean(pred)
        target_mean = target - torch.mean(target)
        
        # Calculate covariance and variances
        cov = torch.sum(pred_mean * target_mean)
        var_pred = torch.sum(pred_mean ** 2)
        var_target = torch.sum(target_mean ** 2)
        
        # Add epsilon to prevent division by zero
        corr = cov / (torch.sqrt(var_pred * var_target) + 1e-8)
        
        # We return the negative squared correlation
        # The Bias Predictor minimizes this (maximizing correlation)
        # The Feature Extractor (via GRL) maximizes it (driving correlation to 0)
        return -(corr ** 2)



# ==========================================
# 3. TRAINING AND EVALUATION LOOP
# ==========================================
# 1. Use Gaussian Noise instead of Spatial Flips
def add_gaussian_noise(tensor, std=0.05):
    return tensor + torch.randn(tensor.size()).to(tensor.device) * std



# 1. Update the Training Augmentations (Per Page 10 of the paper)
# Since your data is 32x32 grayscale, we adapt the paper's strategy [cite: 516-519]
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # We omit color jitter/grayscale as your data is already synthetic grayscale
])

def train_and_benchmark():
    results_ACCd = []
    results_BWTd = []
    results_FWTd = []

    for run_seed in [42, 1234, 9999]:
        set_seed(run_seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DiversityEnsembleDANN().to(device)

        criterion_class = nn.CrossEntropyLoss()
        criterion_conf = SquaredCorrelationLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_stages = 5
        A_i = 0.75 
        R = np.zeros((num_stages, num_stages))
        epochs = 30
        step_size = 0.125 

        # --- 1. PRE-GENERATE TEST SETS (Single View as per Paper ) ---
        test_loaders = []
        for stage in range(num_stages):
            scale_shift = stage * step_size
            cf_val, _, x_val, y_val = generate_data(N=500, seed=1235, scale=scale_shift)
            test_dataset = SyntheticDataset(x_val, y_val, cf_val)
            test_loaders.append(DataLoader(test_dataset, batch_size=128))

        # --- 2. THE TWO-VIEW TRAINING LOOP ---
        for stage in range(num_stages):
            print(f"\n--- Training on Stage {stage + 1} ---")
            scale_shift = stage * step_size
            cf_train, _, x_train, y_train = generate_data(N=2048, seed=1234, scale=scale_shift)
            train_dataset = SyntheticDataset(x_train, y_train, cf_train)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

            # Hyperparameters
            lambda_dann = 5.0    
            lambda_div = 0.5     
            alpha_margin = 1.0   

            model.train()

            for epoch in range(epochs): 
                for i, batch in enumerate(train_loader):
                    X_orig = batch['image'].to(device).float().permute(0, 3, 1, 2)
                    y_batch = batch['label'].to(device).long()
                    conf_batch = batch['cfs'].to(device).float()
                    
                    # --- NEW TWO-VIEW GENERATION (Safe for Blobs) ---
                    # View 1 and View 2 are the same blobs, just with different noise
                    X1 = add_gaussian_noise(X_orig, std=0.05)
                    X2 = add_gaussian_noise(X_orig, std=0.10) # Slightly noisier view
                    
                    # --- ADJUST HYPERPARAMETERS ---
                    # Because we have TWO views, the adversary (DANN) is twice as present.
                    # Lower lambda_dann to compensate.
                    lambda_dann = 2.0  # Lowered from 5.0
                    lambda_div = 0.5
                    
                    # Dynamic alpha for DANN
                    p = float(i + epoch * len(train_loader)) / (epochs * len(train_loader))
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    
                    optimizer.zero_grad()
                    
                    # Forward pass for both views [cite: 83, 84]
                    preds1, confs1, subnets1 = model(X1, alpha=alpha)
                    preds2, confs2, subnets2 = model(X2, alpha=alpha)
                    
                    # 1. Classification Loss (Average of both views) [cite: 91, 92]
                    loss_class = (criterion_class(preds1, y_batch) + criterion_class(preds2, y_batch)) / 2
                    
                    # 2. Confounder Loss (Average of both views)
                    loss_conf = (criterion_conf(confs1, conf_batch) + criterion_conf(confs2, conf_batch)) / 2
                    
                    # 3. Diversity Loss (Summed across both views as per Eq 2 )
                    std1 = torch.sqrt(subnets1.var(dim=1) + 1e-8).mean()
                    std2 = torch.sqrt(subnets2.var(dim=1) + 1e-8).mean()
                    
                    loss_div = torch.clamp(alpha_margin - std1, min=0.0) + \
                            torch.clamp(alpha_margin - std2, min=0.0)
                    
                    current_lambda_div = lambda_div * alpha
                    
                    # Total Combined Loss [cite: 119]
                    loss = loss_class + (lambda_dann * loss_conf) + (current_lambda_div * loss_div) 
                    
                    loss.backward()
                    optimizer.step()
                    
            # --- 3. EVALUATION LOOP (Single View Only ) ---
            model.eval()
            with torch.no_grad():
                for eval_stage in range(num_stages):
                    correct, total = 0, 0
                    for batch in test_loaders[eval_stage]:
                        X_batch = batch['image'].to(device).float().permute(0, 3, 1, 2)
                        y_batch = batch['label'].to(device).long()
                        preds, _, _ = model(X_batch)
                        correct += (preds.argmax(1) == y_batch).sum().item()
                        total += y_batch.size(0)
                    R[stage][eval_stage] = correct / total
                    print(f"  Accuracy on Stage {eval_stage + 1}: {R[stage][eval_stage]:.4f}")

        # ==========================================
        # CALCULATE METRICS FOR THIS SEED (INSIDE THE LOOP)
        # ==========================================
        ACCd = np.mean([abs(R[num_stages-1][i] - A_i) for i in range(num_stages)])
        BWTd = np.mean([abs(R[num_stages-1][i] - A_i) - abs(R[i][i] - A_i) for i in range(num_stages - 1)])
        FWTd = np.mean([abs(R[i-1][i] - A_i) for i in range(1, num_stages)])
        
        results_ACCd.append(ACCd)
        results_BWTd.append(BWTd)
        results_FWTd.append(FWTd)

    # ==========================================
    # FINAL AVERAGES (OUTSIDE THE LOOP)
    # ==========================================
    print("\n=== FINAL BENCHMARK METRICS (Across 3 Seeds) ===")
    print(f"Final ACCd: {np.mean(results_ACCd):.4f} ± {np.std(results_ACCd):.4f}")
    print(f"Final BWTd: {np.mean(results_BWTd):.4f} ± {np.std(results_BWTd):.4f}")
    print(f"Final FWTd: {np.mean(results_FWTd):.4f} ± {np.std(results_FWTd):.4f}")

if __name__ == "__main__":
    train_and_benchmark()