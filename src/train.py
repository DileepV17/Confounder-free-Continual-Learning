"""
This file reuses some of the code from https://github.com/mlu355/MetadataNorm  
"""

import argparse
import os
import dcor
import numpy as np
import torch
import torch.utils.data
from dataset import generate_data
from synthetic_dataset import SyntheticDataset
from utils import *
from conv import Conv
import random

# define accuracy function
def binary_acc(y_pred, y_true):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc


parser = argparse.ArgumentParser(description='Trainer for Recursive Metadata Normalization')
parser.add_argument('--debias', action=argparse.BooleanOptionalAction, default=True, help='Learn confounder-invariant features?')
parser.add_argument('--save_dir', type=str, default='runs/', help='directory name to store output')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--N', type=int, default=1024, help='size of each group (group A or group B)')
parser.add_argument('--runs', type=int, default=5, help='number of experimental runs')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--forgetting_factor', type=float, default=0.999, help='forgetting factor')
parser.add_argument('--reg', type=float, default=1e-4, help='regularization strength')
parser.add_argument('--step_size', type=int, default=20, help='step size for scheduler')
parser.add_argument('--gamma', type=float, default=0.8, help='gamma for scheduler')
args = parser.parse_args()


def run_experiment(model,
                   experiment_name,
                   batch_size,
                   learning_rate,
                   debias,
                   forgetting_factor,
                   reg,
                   stage,
                   x,
                   labels,
                   cf,
                   x_val,
                   labels_val,
                   cf_val,
                   epochs,
                   N,
                   step_size,
                   gamma):
    
    trainset_size = 2 * N
    stage_name = os.path.join(experiment_name, f'stage{stage}')
    log_file = os.path.join(stage_name, 'metrics.txt')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create model
    if model is None:
        model = Conv(
            debias=debias,
            forgetting_factor=forgetting_factor,
            reg=reg
        )
    else:
        print('continuing model training')
   
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    iterations = 2 * N // batch_size
    print(model)
    
    # Make dataloaders
    print('Making dataloaders...')
    train_set = SyntheticDataset(x, labels, cf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_set = SyntheticDataset(x_val, labels_val, cf_val)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Run training 
    acc_history = []
    acc_history_val = []
    dc0s_val = []
    dc1s_val = []
    losses = []
    losses_val = []
        
    for e in range(epochs):
        cfs_val = []
        feature_val = []
        epoch_acc = 0
        epoch_acc_val = 0
        epoch_loss = 0
        epoch_loss_val = 0
        pred_vals = []
        target_vals = []

        # Training pass
        model = model.train()
        for i, sample_batched in enumerate(train_loader):
            data = sample_batched['image'].float()
            target = sample_batched['label'].float()
            cf_batch = sample_batched['cfs'].float()
            data, target = data.to(device), target.to(device)

            # Use both confounders and labels for estimating regression coefficients
            B = target.shape[0]
            X_batch = np.zeros((B,3))
            X_batch[:,0] = target.cpu().detach().numpy()
            X_batch[:,1] = cf_batch.cpu().detach().numpy()
            X_batch[:,2] = np.ones((B,))
            X_batch = torch.Tensor(X_batch).to(device)

            # Forward pass
            optimizer.zero_grad()
            y_pred, fc = model(data, X_batch)
            loss = criterion(y_pred.squeeze(), target)
            acc = binary_acc(y_pred.squeeze(), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # Validation pass
        model = model.eval()
        for _, sample_batched in enumerate(val_loader):
            data = sample_batched['image'].float()
            target = sample_batched['label'].float()
            cf_batch = sample_batched['cfs'].float()
            data, target = data.to(device), target.to(device)

            B = target.shape[0]
            X_batch = np.zeros((B,3))
            X_batch[:,0] = target.cpu().detach().numpy()
            X_batch[:,1] = cf_batch.cpu().detach().numpy()
            X_batch[:,2] = np.ones((B,))

            with torch.no_grad():
                X_batch = torch.Tensor(X_batch).to(device)
                y_pred, fc = model(data, X_batch)
                loss = criterion(y_pred, target.unsqueeze(1))
                acc = binary_acc(y_pred, target.unsqueeze(1))
                epoch_loss_val += loss.item()
                epoch_acc_val += acc.item()

            # Save learned features
            feature_val.append(fc)
            cfs_val.append(cf_batch)
            target_vals.append(target.cpu())
            pred_vals.append(y_pred.cpu())
            
        # Calculate distance correlation between confounders and learned features
        epoch_targets = np.concatenate(target_vals, axis=0)
        i0_val = np.where(epoch_targets == 0)[0]
        i1_val = np.where(epoch_targets == 1)[0]
        epoch_layer = np.concatenate(feature_val, axis=0)
        epoch_cf = np.concatenate(cfs_val, axis=0)
        np.save(os.path.join(stage_name, "features.npy"), epoch_layer)
        np.save(os.path.join(stage_name, "cfs.npy"), epoch_cf)
        dc0_val = dcor.u_distance_correlation_sqr(epoch_layer[i0_val], epoch_cf[i0_val])
        dc1_val = dcor.u_distance_correlation_sqr(epoch_layer[i1_val], epoch_cf[i1_val])
        print('correlations for feature 0:', dc0_val)
        print('correlations for feature 1:', dc1_val)
        dc0s_val.append(dc0_val)
        dc1s_val.append(dc1_val)

        curr_acc = epoch_acc / (2*N)
        acc_history.append(curr_acc)
        losses.append(epoch_loss / iterations)
        curr_acc_val = epoch_acc_val / (2*N)
        acc_history_val.append(curr_acc_val)
        losses_val.append(epoch_loss_val / iterations)

        print('learning rate:', optimizer.param_groups[0]['lr'])
        lr_scheduler.step()
            
        print(f'Train: Epoch {e+0:03}: | Loss: {epoch_loss/iterations:.5f} | Acc: {epoch_acc / (2*N):.3f}')
        print(f'Val: Epoch {e+0:03}: | Loss: {epoch_loss_val/iterations:.5f} | Acc: {epoch_acc_val / (2*N):.3f}')
    
        with open(log_file, 'a') as f:
            f.write(str(e) + '\t' + str(curr_acc_val) + '\t' + str(dc0_val) + '\t' + str(dc1_val) + '\t' + str(epoch_loss_val) + '\n')

    torch.save(model.state_dict(), os.path.join(stage_name, 'best_model.pth'))
    return model


def test(
    model,
    run_name_base,
    batch_size,
    stage_trained,
    stage_tested,
    x,
    labels,
    cf,
    N,
):
    criterion = torch.nn.BCELoss()
    
    # Make dataloaders
    val_set = SyntheticDataset(x, labels, cf)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfs_val = []
    feature_val = []
    epoch_acc_val = 0
    epoch_loss_val = 0
    pred_vals = []
    target_vals = []

    # Validation pass
    model = model.eval()
    for _, sample_batched in enumerate(val_loader):
        data = sample_batched['image'].float()
        target = sample_batched['label'].float()
        cf_batch = sample_batched['cfs'].float()
        data, target = data.to(device), target.to(device)

        B = target.shape[0]
        X_batch = np.zeros((B,3))
        X_batch[:,0] = target.cpu().detach().numpy()
        X_batch[:,1] = cf_batch.cpu().detach().numpy()
        X_batch[:,2] = np.ones((B,))

        with torch.no_grad():
            X_batch = torch.Tensor(X_batch).to(device)
            y_pred, fc = model(data, X_batch)
            loss = criterion(y_pred, target.unsqueeze(1))
            acc = binary_acc(y_pred, target.unsqueeze(1))
            epoch_loss_val += loss.item()
            epoch_acc_val += acc.item()

        # Save learned features
        feature_val.append(fc)
        cfs_val.append(cf_batch)
        target_vals.append(target.cpu())
        pred_vals.append(y_pred.cpu())
        
    # Calculate distance correlation between confounders and learned features
    epoch_targets = np.concatenate(target_vals, axis=0)
    i0_val = np.where(epoch_targets == 0)[0]
    i1_val = np.where(epoch_targets == 1)[0]
    epoch_layer = np.concatenate(feature_val, axis=0)
    epoch_cf = np.concatenate(cfs_val, axis=0)
    np.save(os.path.join(run_name_base, f"{stage_trained}_{stage_tested}_features.npy"), epoch_layer)
    np.save(os.path.join(run_name_base, f"{stage_trained}_{stage_tested}_cfs.npy"), epoch_cf)
    dc0_val = dcor.u_distance_correlation_sqr(epoch_layer[i0_val], epoch_cf[i0_val])
    dc1_val = dcor.u_distance_correlation_sqr(epoch_layer[i1_val], epoch_cf[i1_val])
        
    print(f'Stage Trained: {stage_trained}| Stage Tested: {stage_tested} | Acc: {epoch_acc_val / (2 * N):.5f} | dcor0: {dc0_val} | dcor1: {dc1_val} | Loss: {epoch_loss_val/(2 * N / batch_size):.5f}')

    with open(os.path.join(run_name_base, 'metrics.txt'), 'a') as f:
        f.write(str(stage_trained) + '\t' + str(stage_tested) + '\t' + str(epoch_acc_val / (2 * N)) + '\t' + str(dc0_val) + '\t' + str(dc1_val) + '\t' + str(epoch_loss_val/(2 * N / batch_size)) + '\n')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_indices(length, frac):
    num_samples = int(frac * length)
    random_indices = np.random.choice(length, num_samples, replace=False)
    return random_indices


def run_experiments():
    N = args.N
    runs = args.runs

    # Generate training and validation data
    cf1, _, x1, y = generate_data(N, seed=args.seed, scale=0)
    cf2, _, x2, _ = generate_data(N, seed=args.seed, scale=0.125)
    cf3, _, x3, _ = generate_data(N, seed=args.seed, scale=0.25)
    cf4, _, x4, _ = generate_data(N, seed=args.seed, scale=0.375)
    cf5, _, x5, _ = generate_data(N, seed=args.seed, scale=0.5)

    cf1_val, _, x1_val, y_val = generate_data(N, seed=args.seed + 1, scale=0)
    cf2_val, _, x2_val, _ = generate_data(N, seed=args.seed + 1, scale=0.125)
    cf3_val, _, x3_val, _ = generate_data(N, seed=args.seed + 1, scale=0.25)
    cf4_val, _, x4_val, _ = generate_data(N, seed=args.seed + 1, scale=0.375)
    cf5_val, _, x5_val, _ = generate_data(N, seed=args.seed + 1, scale=0.5)

    stages = [
        (cf1, x1, y, cf1_val, x1_val, y_val),
        (cf2, x2, y, cf2_val, x2_val, y_val),
        (cf3, x3, y, cf3_val, x3_val, y_val),
        (cf4, x4, y, cf4_val, x4_val, y_val),
        (cf5, x5, y, cf5_val, x5_val, y_val),
    ]

    tests = [
        (cf1_val, x1_val, y_val),
        (cf2_val, x2_val, y_val),
        (cf3_val, x3_val, y_val),
        (cf4_val, x4_val, y_val),
        (cf5_val, x5_val, y_val)
    ]

    for run in range(1, runs+1):
        batch_size = args.batch_size
        debias = args.debias
        method = 'baseline' if not debias else 'rmdn'
        learning_rate = args.lr
        epochs = args.epochs
        forgetting_factor = args.forgetting_factor
        reg = args.reg
        step_size = args.step_size
        gamma = args.gamma
    
        set_seed(run)
        
        run_name_base = os.path.join(args.save_dir, method)
        experiment_name = os.path.join(run_name_base, 'e' + str(epochs) + '_lr' + str(learning_rate) + '_reg' + str(reg), f'run{run}/')
    
        run_log_file = os.path.join(experiment_name, 'metrics.txt')
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        with open(run_log_file, 'w') as f:
            f.write('stage_trained' + '\t' + 'stage_tested' + '\t' + 'acc' + '\t' + 'dc0_val' + '\t' + 'dc1_val' + '\t' + 'loss' + '\n')
        
        model = None
    
        for stage, (cf, x, y, cf_val, x_val, y_val) in enumerate(stages):
            # Run experiments
            stage_name = os.path.join(experiment_name, f'stage{stage}')
            log_file = os.path.join(stage_name, 'metrics.txt')
            if not os.path.exists(stage_name):
                os.makedirs(stage_name)
            print(stage_name)
    
            with open(log_file, 'w') as f:
                f.write('method=' + method + '\t' 
                        + 'batch_size=' + str(batch_size) + '\t' 
                        + 'forgetting_factor=' + str(forgetting_factor) + '\t' 
                        + 'reg=' + str(reg) + '\t'
                        + 'lr=' + str(learning_rate) + '\t'
                        + 'step_size=' + str(step_size) + '\t'
                        + 'gamma=' + str(gamma) + '\t'
                        + 'N='  + str(N) + '\n')
                f.write('epoch' + '\t' + 'acc' + '\t' + 'dc0_val' + '\t' + 'dc0_val' + '\t' + 'loss' + '\n')
    
            print('\nRunning experiment ' + method + ': Stage ' + str(stage))
            print('-----------------------------------------------------------')
            model = run_experiment(
                model=model,
                experiment_name=experiment_name,
                batch_size=batch_size,
                learning_rate=learning_rate,
                debias=debias,
                forgetting_factor=forgetting_factor,
                reg=reg,
                stage=stage,
                epochs=epochs,
                x=np.transpose(x, (0, 3, 1, 2)),
                labels=y,
                cf=cf,
                x_val=np.transpose(x_val, (0, 3, 1, 2)),
                labels_val=y_val,
                cf_val=cf_val,
                N=N,
                step_size=step_size,
                gamma=gamma
            )

            for j, (cf_val, x_val, y) in enumerate(tests):
                test(
                    model=model,
                    run_name_base=experiment_name,
                    batch_size=batch_size,
                    stage_trained=stage,
                    stage_tested=j,
                    x=np.transpose(x_val, (0, 3, 1, 2)),
                    labels=y,
                    cf=cf_val,
                    N=N
                )


if __name__ == '__main__':
    run_experiments()



import os
import numpy as np
import matplotlib.pyplot as plt

def parse_rmdn_metrics():
    """Parses the authors' output files and calculates CL metrics."""
    rmdn_accd, rmdn_bwtd, rmdn_fwtd = [], [], []
    A_i = 0.75
    num_stages = 5
    
    # The authors' script defaults to runs 1, 2, and 3
    for run in range(1, 4):
        filepath = f"runs/rmdn/e100_lr0.0001_reg0.0001/run{run}/metrics.txt"
        
        if not os.path.exists(filepath):
            print(f"ERROR: Cannot find {filepath}. Did you run train.py first?")
            return None, None
            
        R = np.zeros((num_stages, num_stages))
        with open(filepath, 'r') as f:
            lines = f.readlines()[1:] # Skip the header
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    st_train = int(parts[0])
                    st_test = int(parts[1])
                    acc = float(parts[2])
                    R[st_train, st_test] = acc
                    
        # Calculate metrics for this run
        ACCd = np.mean([abs(R[num_stages-1][i] - A_i) for i in range(num_stages)])
        BWTd = np.mean([abs(R[num_stages-1][i] - A_i) - abs(R[i][i] - A_i) for i in range(num_stages - 1)])
        FWTd = np.mean([abs(R[i-1][i] - A_i) for i in range(1, num_stages)])
        
        rmdn_accd.append(ACCd)
        rmdn_bwtd.append(BWTd)
        rmdn_fwtd.append(FWTd)
        
    means = [np.mean(rmdn_accd), np.mean(rmdn_bwtd), np.mean(rmdn_fwtd)]
    stds = [np.std(rmdn_accd), np.std(rmdn_bwtd), np.std(rmdn_fwtd)]
    return means, stds



