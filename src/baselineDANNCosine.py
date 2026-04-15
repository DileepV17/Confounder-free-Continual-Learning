import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Function
from torch.utils.data import DataLoader
from dataset import generate_data
from synthetic_dataset import SyntheticDataset
import random
import os

# ==========================================
# REPRODUCIBILITY SETUP
# ==========================================
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


# ==========================================
# 2. DANN ARCHITECTURE (BR-NET EQUIVALENT)
# ==========================================
class SquaredCorrelationLoss(nn.Module):
    def forward(self, pred, target):
        # Subtract the mean
        pred_mean = pred - torch.mean(pred)
        target_mean = target - torch.mean(target)
        
        # Calculate covariance and variances
        cov = torch.sum(pred_mean * target_mean)
        var_pred = torch.sum(pred_mean ** 2)
        var_target = torch.sum(target_mean ** 2)
        
        # Add epsilon to prevent division by zero
        corr = cov / (torch.sqrt(var_pred * var_target) + 1e-8)
        
        # We return NEGATIVE squared correlation. 
        # The BP network minimizes this (maximizing correlation).
        # The GRL reverses it, so the Feature Extractor maximizes it (driving correlation to 0).
        return -(corr ** 2)

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

# ==========================================
# 3. TRAINING AND EVALUATION LOOP
# ==========================================
def train_and_benchmark():
    results_ACCd = []
    results_BWTd = []
    results_FWTd = []

    for run_seed in [42, 1234, 9999]:
        set_seed(run_seed)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DANN_Continual().to(device)
        
        criterion_class = nn.CrossEntropyLoss()
        criterion_conf = SquaredCorrelationLoss() 


# Lower LR to 0.0001 and add weight decay (L2 regularization)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        
        # Add the Cosine Annealing Scheduler
        epochs = 100
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        num_stages = 5
        A_i = 0.75 
        R = np.zeros((num_stages, num_stages))
        
        # The paper's scaling factor for distribution shifts
        step_size = 0.125 

        # --- 1. PRE-GENERATE THE PAPER'S TEST SETS ---
        test_loaders = []
        for stage in range(num_stages):
            scale_shift = stage * step_size
            
            # FIXED UNPACKING AND SEED (1235 for validation)
            cf_val, _, x_val, y_val = generate_data(N=500, seed=1235, scale=scale_shift)
            
            # Wrap it in their exact dataset class (NORMAL ORDER)
            test_dataset = SyntheticDataset(x_val, y_val, cf_val)
            test_loaders.append(DataLoader(test_dataset, batch_size=128))

        # --- 2. THE TRAINING LOOP ---
        for stage in range(num_stages):
            print(f"\n--- Training on Stage {stage + 1} ---")
            scale_shift = stage * step_size
            
            # FIXED UNPACKING AND SEED (1234 for training)
            cf_train, _, x_train, y_train = generate_data(N=2048, seed=1234, scale=scale_shift)
            
            train_dataset = SyntheticDataset(x_train, y_train, cf_train)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            # Set baseline adversarial penalty 
            # (Using 5.0 to be comparable with the ensemble parameter)
            lambda_dann = 5.0 

            model.train()
            for epoch in range(epochs): 
                for i, batch in enumerate(train_loader):
                    # Pull from dictionary with proper types
                    X_batch = batch['image'].to(device).float()
                    y_batch = batch['label'].to(device).long()
                    conf_batch = batch['cfs'].to(device).float()
                    
                    # CHANGE SHAPE: [128, 32, 32, 1] -> [128, 1, 32, 32]
                    X_batch = X_batch.permute(0, 3, 1, 2)
                    
                    # Dynamic alpha for DANN warm-up
                    p = float(i + epoch * len(train_loader)) / (epochs * len(train_loader))
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    
                    optimizer.zero_grad()
                    
                    class_preds, conf_preds = model(X_batch, alpha=alpha)
                    loss_class = criterion_class(class_preds, y_batch)
                    loss_conf = criterion_conf(conf_preds, conf_batch)
                    
                    loss = loss_class + (lambda_dann * loss_conf) 
                    
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                                    
    # --- 3. EVALUATION LOOP (Single View Only ) ---
            model.eval()
            with torch.no_grad():
                for eval_stage in range(num_stages):
                    correct, total = 0, 0
                    for batch in test_loaders[eval_stage]:
                        X_batch = batch['image'].to(device).float().permute(0, 3, 1, 2)
                        y_batch = batch['label'].to(device).long()
                        preds, _, = model(X_batch)
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