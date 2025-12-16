"""
Measure the learning speed of different model architectures on random data.

Learning speed is defined as the number of training steps required to saturate
model memory (reach near-perfect accuracy on random data). By comparing
saturation steps across different dataset sizes and model architectures,
we can estimate learning speed in steps per bit.
"""

import argparse
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import TransformerTorch
from data import random_target_data_torch

import matplotlib.pyplot as plt
import seaborn as sns


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SpeedTrainer:
    """
    Trainer for measuring learning speed (steps to saturation).
    
    Trains until accuracy reaches a saturation threshold and counts
    the total number of training steps.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        batch_size: int = 512,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        
        # Traces
        self.train_loss_trace = []
        self.train_acc_trace = []
        self.steps_trace = []  # Track cumulative steps at each epoch
    
    def _make_batches(self, X: torch.Tensor, T: torch.Tensor):
        """Yield batches from data."""
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i+bs], T[i:i+bs]
    
    def train(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        max_epochs: int = 10000,
        saturation_threshold: float = 99.5,
        patience: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Train until accuracy saturates (reaches threshold).
        
        Args:
            train_data: Tuple of (X, T) tensors
            max_epochs: Maximum number of epochs
            saturation_threshold: Accuracy threshold to consider saturated (%)
            patience: Number of epochs to confirm saturation
            verbose: Whether to show progress bar
        
        Returns:
            Dictionary with training results including steps to saturation
        """
        X_train, T_train = train_data
        n_samples = X_train.shape[0]
        
        # Calculate number of steps per epoch
        bs = self.batch_size if self.batch_size != -1 else n_samples
        steps_per_epoch = (n_samples + bs - 1) // bs
        
        total_steps = 0
        saturation_step = None
        epochs_above_threshold = 0
        
        epoch_iter = tqdm(range(max_epochs), desc='Training', unit='epoch') if verbose else range(max_epochs)
        
        for epoch in epoch_iter:
            self.model.train()
            
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_shuffled = X_train[perm]
            T_shuffled = T_train[perm]
            
            total_loss = 0.0
            total_correct = 0
            
            for Xb, Tb in self._make_batches(X_shuffled, T_shuffled):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(Xb)
                loss = F.cross_entropy(logits, Tb)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * Xb.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == Tb).sum().item()
                
                total_steps += 1
            
            avg_loss = total_loss / n_samples
            avg_acc = (total_correct / n_samples) * 100  # Convert to percentage
            
            self.train_loss_trace.append(avg_loss)
            self.train_acc_trace.append(avg_acc)
            self.steps_trace.append(total_steps)
            
            if verbose:
                epoch_iter.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.2f}%',
                    'steps': total_steps
                })
            
            # Check for saturation
            if avg_acc >= saturation_threshold:
                epochs_above_threshold += 1
                if saturation_step is None:
                    saturation_step = total_steps
                
                # Confirm saturation after patience epochs
                if epochs_above_threshold >= patience:
                    if verbose:
                        print(f"\nSaturation reached at step {saturation_step} "
                              f"(epoch {epoch + 1 - patience}), acc={avg_acc:.2f}%")
                    break
            else:
                epochs_above_threshold = 0
                saturation_step = None
        
        # If we didn't reach saturation, use final values
        if saturation_step is None:
            saturation_step = total_steps
            if verbose:
                print(f"\nDid not reach saturation threshold ({saturation_threshold}%). "
                      f"Final acc: {avg_acc:.2f}%")
        
        return {
            'epochs_trained': epoch + 1,
            'total_steps': total_steps,
            'saturation_step': saturation_step,
            'final_loss': avg_loss,
            'final_acc': avg_acc,
            'train_loss_trace': np.array(self.train_loss_trace),
            'train_acc_trace': np.array(self.train_acc_trace),
            'steps_trace': np.array(self.steps_trace),
            'saturated': avg_acc >= saturation_threshold
        }


def run_speed_experiment(
    n_samples: int,
    dim: int,
    depth: int,
    heads: int,
    p: int,
    max_epochs: int,
    saturation_threshold: float,
    patience: int,
    args,
    verbose: bool = True
) -> Dict:
    """
    Run a single speed experiment with given dataset size and model config.
    
    Returns:
        Dictionary with experiment results
    """
    n_tokens = p + 2  # Full vocabulary size (p digits + operator + equals)
    
    # Generate random target data
    X_train, T_train = random_target_data_torch(n_samples, p, seq_len=4, device='cpu')
    
    # Build model
    model_kwargs = {
        'depth': depth,
        'dim': dim,
        'heads': heads,
        'n_tokens': n_tokens,
        'seq_len': 4,
        'dropout': args.dropout
    }
    
    # Select device
    if args.device is not None:
        device = args.device
    elif args.cpu:
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    model = TransformerTorch(**model_kwargs).to(device)
    param_count = count_parameters(model)
    
    if verbose:
        print(f"  Dataset size: {n_samples}, Model dim: {dim}, Parameters: {param_count:,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    trainer = SpeedTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=args.batch_size,
        device=device
    )
    
    results = trainer.train(
        (X_train, T_train),
        max_epochs=max_epochs,
        saturation_threshold=saturation_threshold,
        patience=patience,
        verbose=verbose
    )
    
    # Add experiment metadata
    results['n_samples'] = n_samples
    results['dim'] = dim
    results['depth'] = depth
    results['heads'] = heads
    results['param_count'] = param_count
    results['p'] = p
    
    # Compute dataset bits
    bits_per_example = np.log2(n_tokens)
    results['dataset_bits'] = n_samples * bits_per_example
    results['bits_per_example'] = bits_per_example
    
    return results


def save_results(results: Dict, args):
    """Save individual experiment results."""
    os.makedirs(args.data_dir, exist_ok=True)
    
    fname = os.path.join(
        args.data_dir,
        f'speed_dim{results["dim"]}_samples{results["n_samples"]}.npz'
    )
    
    np.savez(
        fname,
        n_samples=results['n_samples'],
        dim=results['dim'],
        depth=results['depth'],
        heads=results['heads'],
        param_count=results['param_count'],
        p=results['p'],
        epochs_trained=results['epochs_trained'],
        total_steps=results['total_steps'],
        saturation_step=results['saturation_step'],
        final_loss=results['final_loss'],
        final_acc=results['final_acc'],
        dataset_bits=results['dataset_bits'],
        bits_per_example=results['bits_per_example'],
        saturated=results['saturated'],
        train_loss_trace=results['train_loss_trace'],
        train_acc_trace=results['train_acc_trace'],
        steps_trace=results['steps_trace']
    )
    
    return fname


def load_or_run_experiment(
    n_samples: int,
    dim: int,
    args,
    force: bool = False,
    verbose: bool = True
) -> Dict:
    """Load existing results or run a new experiment."""
    fname = os.path.join(
        args.data_dir,
        f'speed_dim{dim}_samples{n_samples}.npz'
    )
    
    if os.path.exists(fname) and not force:
        if verbose:
            print(f"  Loading existing results: {fname}")
        data = np.load(fname)
        return {key: data[key].item() if data[key].ndim == 0 else data[key] 
                for key in data.files}
    
    # Run experiment
    result = run_speed_experiment(
        n_samples=n_samples,
        dim=dim,
        depth=args.depth,
        heads=args.heads,
        p=args.p,
        max_epochs=args.epochs,
        saturation_threshold=args.saturation_threshold,
        patience=args.patience,
        args=args,
        verbose=verbose
    )
    
    # Save results
    save_results(result, args)
    
    return result


def plot_learning_speed_curves(
    all_results: Dict[int, List[Dict]],
    p: int = 97,
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict[int, Tuple[float, float]]:
    """
    Plot saturation steps vs dataset size (in bits) for different model sizes.
    
    Args:
        all_results: Dict mapping dimension to list of result dicts
        p: Prime number for computing bits
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    
    Returns:
        speed_estimates: Dict mapping param_count to (slope, intercept) of linear fit
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort dims for consistent legend ordering
    dims = sorted(all_results.keys())
    
    # Get colors from seaborn crest colormap
    colors = sns.color_palette("crest", n_colors=len(dims))
    
    speed_estimates = {}
    
    for idx, dim in enumerate(dims):
        results = all_results[dim]
        
        # Filter to only saturated experiments
        saturated_results = [r for r in results if r.get('saturated', True)]
        if not saturated_results:
            print(f"Warning: No saturated results for dim={dim}")
            continue
        
        # Sort by dataset size
        saturated_results = sorted(saturated_results, key=lambda x: x['n_samples'])
        
        dataset_bits = [r['dataset_bits'] for r in saturated_results]
        saturation_steps = [r['saturation_step'] for r in saturated_results]
        param_count = saturated_results[0]['param_count']
        
        # Format parameter count for legend
        if param_count >= 1e6:
            param_str = f'{param_count/1e6:.1f}M'
        elif param_count >= 1e3:
            param_str = f'{param_count/1e3:.0f}K'
        else:
            param_str = str(param_count)
        
        ax.plot(
            dataset_bits,
            saturation_steps,
            marker='o',
            markersize=8,
            linewidth=2,
            color=colors[idx],
            label=f'{param_str} params (dim={dim})'
        )
        
        # Fit linear model for speed estimate
        if len(dataset_bits) >= 2:
            slope, intercept = np.polyfit(dataset_bits, saturation_steps, 1)
            speed_estimates[param_count] = (slope, intercept, dim)
    
    ax.set_xlabel('Dataset Size (bits)', fontsize=14)
    ax.set_ylabel('Steps to Saturation', fontsize=14)
    ax.set_title('Learning Speed: Steps to Memorize Random Data', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Legend
    ax.legend(
        title='Model Size',
        loc='upper left',
        fontsize=10,
        title_fontsize=11
    )
    
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved learning speed plot: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return speed_estimates


def plot_speed_vs_model_size(
    speed_estimates: Dict[int, Tuple[float, float, int]],
    save_path: Optional[str] = None,
    show: bool = True
) -> Tuple[float, float, float]:
    """
    Plot learning speed (steps per bit) vs model parameter count.
    
    Args:
        speed_estimates: Dict mapping param_count to (slope, intercept, dim)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    
    Returns:
        Tuple of (slope_of_log_fit, intercept_of_log_fit, r_squared)
    """
    if len(speed_estimates) < 2:
        print("Not enough data points for speed vs model size plot")
        return 0.0, 0.0, 0.0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by parameter count
    sorted_items = sorted(speed_estimates.items())
    param_counts = np.array([p for p, _ in sorted_items])
    slopes = np.array([s[0] for _, s in sorted_items])  # steps per bit
    dims = [s[2] for _, s in sorted_items]
    
    # Use crest colormap
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    
    scatter = ax.scatter(
        param_counts, slopes, 
        c=dims, cmap=crest_cmap,
        s=100, alpha=0.8, edgecolors='black', linewidths=1
    )
    
    # Add labels for each point
    for pc, slope, dim in zip(param_counts, slopes, dims):
        ax.annotate(f'dim={dim}', (pc, slope), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    # Fit power law: slope = a * params^b  =>  log(slope) = log(a) + b * log(params)
    log_params = np.log10(param_counts)
    log_slopes = np.log10(slopes)
    
    b, log_a = np.polyfit(log_params, log_slopes, 1)
    a = 10 ** log_a
    
    # Calculate R²
    y_pred = log_a + b * log_params
    ss_res = np.sum((log_slopes - y_pred) ** 2)
    ss_tot = np.sum((log_slopes - np.mean(log_slopes)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Plot fitted curve
    x_fit = np.logspace(np.log10(param_counts.min() * 0.5), 
                        np.log10(param_counts.max() * 2), 100)
    y_fit = a * x_fit ** b
    ax.plot(x_fit, y_fit, '--', color='red', linewidth=2, alpha=0.7,
            label=f'Fit: speed = {a:.2e} × params^{b:.2f}')
    
    ax.set_xlabel('Model Parameters', fontsize=14)
    ax.set_ylabel('Learning Speed (steps per bit)', fontsize=14)
    ax.set_title('Learning Speed vs Model Size', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Dimension')
    cbar.ax.tick_params(labelsize=11)
    
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add text box with stats
    textstr = f'Power law exponent: {b:.3f}\n'
    textstr += f'Coefficient: {a:.2e}\n'
    textstr += f'R²: {r_squared:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved speed vs model size plot: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return b, log_a, r_squared


def plot_combined_speed_analysis(
    all_results: Dict[int, List[Dict]],
    p: int = 97,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create a combined figure with both learning speed curves and speed vs model size.
    
    Args:
        all_results: Dict mapping dimension to list of result dicts
        p: Prime number for computing bits
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Sort dims for consistent legend ordering
    dims = sorted(all_results.keys())
    
    # Get colors from seaborn crest colormap
    colors = sns.color_palette("crest", n_colors=len(dims))
    
    speed_estimates = {}
    
    # Left plot: Steps to saturation vs dataset bits
    for idx, dim in enumerate(dims):
        results = all_results[dim]
        
        # Filter to only saturated experiments
        saturated_results = [r for r in results if r.get('saturated', True)]
        if not saturated_results:
            continue
        
        saturated_results = sorted(saturated_results, key=lambda x: x['n_samples'])
        
        dataset_bits = [r['dataset_bits'] for r in saturated_results]
        saturation_steps = [r['saturation_step'] for r in saturated_results]
        param_count = saturated_results[0]['param_count']
        
        if param_count >= 1e6:
            param_str = f'{param_count/1e6:.1f}M'
        elif param_count >= 1e3:
            param_str = f'{param_count/1e3:.0f}K'
        else:
            param_str = str(param_count)
        
        ax1.plot(
            dataset_bits,
            saturation_steps,
            marker='o',
            markersize=8,
            linewidth=2,
            color=colors[idx],
            label=f'{param_str}'
        )
        
        # Fit linear model for speed estimate
        if len(dataset_bits) >= 2:
            slope, intercept = np.polyfit(dataset_bits, saturation_steps, 1)
            speed_estimates[param_count] = (slope, intercept, dim)
    
    ax1.set_xlabel('Dataset Size (bits)', fontsize=14)
    ax1.set_ylabel('Steps to Saturation', fontsize=14)
    ax1.set_title('Learning Speed Curves', fontsize=16, pad=10)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(title='Parameters', loc='upper left', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Right plot: Speed vs model size
    if len(speed_estimates) >= 2:
        sorted_items = sorted(speed_estimates.items())
        param_counts = np.array([p for p, _ in sorted_items])
        slopes = np.array([s[0] for _, s in sorted_items])
        dims_plot = [s[2] for _, s in sorted_items]
        
        crest_cmap = sns.color_palette('crest', as_cmap=True)
        
        scatter = ax2.scatter(
            param_counts, slopes, 
            c=dims_plot, cmap=crest_cmap,
            s=100, alpha=0.8, edgecolors='black', linewidths=1
        )
        
        for pc, slope, dim in zip(param_counts, slopes, dims_plot):
            ax2.annotate(f'{dim}', (pc, slope), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        # Fit power law
        log_params = np.log10(param_counts)
        log_slopes = np.log10(slopes)
        b, log_a = np.polyfit(log_params, log_slopes, 1)
        a = 10 ** log_a
        
        y_pred = log_a + b * log_params
        ss_res = np.sum((log_slopes - y_pred) ** 2)
        ss_tot = np.sum((log_slopes - np.mean(log_slopes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        x_fit = np.logspace(np.log10(param_counts.min() * 0.5), 
                            np.log10(param_counts.max() * 2), 100)
        y_fit = a * x_fit ** b
        ax2.plot(x_fit, y_fit, '--', color='red', linewidth=2, alpha=0.7,
                label=f'speed ∝ params^{b:.2f}')
        
        ax2.set_xlabel('Model Parameters', fontsize=14)
        ax2.set_ylabel('Learning Speed (steps/bit)', fontsize=14)
        ax2.set_title('Speed vs Model Size', fontsize=16, pad=10)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        cbar = plt.colorbar(scatter, ax=ax2, label='Dimension')
        cbar.ax.tick_params(labelsize=11)
        
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        textstr = f'Exponent: {b:.3f}\nR²: {r_squared:.3f}'
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved combined speed analysis plot: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return speed_estimates


def main():
    parser = argparse.ArgumentParser(
        description='Measure learning speed (steps to memorisation) for different architectures'
    )
    
    # Data args
    parser.add_argument('--p', type=int, default=97, 
                        help='Prime number (determines vocabulary size)')
    
    # Model args
    parser.add_argument('--depth', type=int, default=2, help='Transformer depth')
    parser.add_argument('--heads', type=int, default=1, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='Dropout (0.2 to match training)')
    parser.add_argument('--dim-list', type=int, nargs='+', 
                        default=[20, 24, 28],
                        help='List of model dimensions to test')
    
    # Dataset size args
    parser.add_argument('--samples-start', type=int, default=100,
                        help='Starting dataset size')
    parser.add_argument('--samples-end', type=int, default=1000,
                        help='Ending dataset size')
    parser.add_argument('--samples-steps', type=int, default=4,
                        help='Number of dataset sizes to test (log spaced)')
    
    # Optimizer args
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, 
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2')
    
    # Training args
    parser.add_argument('-b', '--batch_size', type=int, default=512, 
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=5000, 
                        help='Maximum epochs')
    parser.add_argument('--saturation-threshold', type=float, default=99.0,
                        help='Accuracy threshold to consider saturated (%%)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Epochs to confirm saturation')
    
    # Output args
    parser.add_argument('--data-dir', type=str, default='data/speed',
                        help='Data output directory')
    parser.add_argument('--plot-dir', type=str, default='media/speed',
                        help='Plot output directory')
    
    # Misc args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU only')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (e.g., "cuda:0", "cpu", "mps")')
    parser.add_argument('--force', action='store_true', 
                        help='Force re-run even if results exist')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (just save)')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories with signature
    signature = f'p{args.p}_seed{args.seed}'
    args.data_dir = os.path.join(args.data_dir, signature)
    args.plot_dir = os.path.join(args.plot_dir, signature)
    
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Generate log-spaced dataset sizes
    dataset_sizes = np.logspace(
        np.log10(args.samples_start),
        np.log10(args.samples_end),
        args.samples_steps
    ).astype(int)
    dataset_sizes = np.unique(dataset_sizes)  # Remove duplicates
    
    n_tokens = args.p + 2
    bits_per_example = np.log2(n_tokens)
    
    print(f"Model dimensions: {args.dim_list}")
    print(f"Dataset sizes: {list(dataset_sizes)}")
    print(f"p: {args.p}")
    print(f"Full vocab size: {n_tokens}")
    print(f"Bits per example: {bits_per_example:.2f}")
    print(f"Saturation threshold: {args.saturation_threshold}%")
    print()
    
    # Run experiments
    all_results = {}
    
    for dim in args.dim_list:
        print(f"\n{'='*60}")
        print(f"Model dimension: {dim}")
        print(f"{'='*60}")
        
        all_results[dim] = []
        
        for n_samples in dataset_sizes:
            print(f"\n  Dataset size: {n_samples}")
            
            result = load_or_run_experiment(
                n_samples=int(n_samples),
                dim=dim,
                args=args,
                force=args.force,
                verbose=True
            )
            
            all_results[dim].append(result)
            
            dataset_bits = n_samples * bits_per_example
            steps_per_bit = result['saturation_step'] / dataset_bits if dataset_bits > 0 else 0
            
            print(f"    Final accuracy: {result['final_acc']:.2f}%")
            print(f"    Saturation steps: {result['saturation_step']:,}")
            print(f"    Dataset bits: {dataset_bits:.0f}")
            print(f"    Steps per bit: {steps_per_bit:.2f}")
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Main learning speed curves plot
    curves_path = os.path.join(args.plot_dir, 'learning_speed_curves.pdf')
    speed_estimates = plot_learning_speed_curves(
        all_results,
        p=args.p,
        save_path=curves_path,
        show=not args.no_show
    )
    
    # Speed vs model size plot
    if speed_estimates:
        speed_path = os.path.join(args.plot_dir, 'speed_vs_model_size.pdf')
        b, log_a, r_squared = plot_speed_vs_model_size(
            speed_estimates,
            save_path=speed_path,
            show=not args.no_show
        )
    
    # Combined analysis plot
    combined_path = os.path.join(args.plot_dir, 'speed_analysis_combined.pdf')
    plot_combined_speed_analysis(
        all_results,
        p=args.p,
        save_path=combined_path,
        show=not args.no_show
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for dim in sorted(all_results.keys()):
        results = all_results[dim]
        param_count = results[0]['param_count']
        
        # Average steps per bit across dataset sizes
        saturated = [r for r in results if r.get('saturated', True)]
        if saturated:
            avg_speed = np.mean([r['saturation_step'] / r['dataset_bits'] 
                                for r in saturated if r['dataset_bits'] > 0])
            print(f"dim={dim:3d}: {param_count:8,} params, "
                  f"avg speed: {avg_speed:.2f} steps/bit")
    
    if speed_estimates:
        print("\n" + "="*60)
        print("LEARNING SPEED ESTIMATES (steps per bit)")
        print("="*60)
        
        for param_count in sorted(speed_estimates.keys()):
            slope, intercept, dim = speed_estimates[param_count]
            print(f"dim={dim:3d}: {param_count:8,} params, "
                  f"slope={slope:.2f} steps/bit, intercept={intercept:.0f} steps")
        
        if len(speed_estimates) >= 2:
            print("\n" + "="*60)
            print("SPEED SCALING")
            print("="*60)
            print(f"Power law exponent: {b:.3f}")
            print(f"R²: {r_squared:.3f}")
            if b < 0:
                print(f"Larger models learn FASTER (fewer steps per bit)")
            else:
                print(f"Larger models learn SLOWER (more steps per bit)")


if __name__ == '__main__':
    main()

