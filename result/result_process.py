import scipy.io
from matplotlib.text import Text
import warnings
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import torch.nn.functional as F
import scipy
import pandas as pd
import os
import glob
import scienceplots
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
plt.style.use('science')
pd.set_option('display.float_format', lambda x: '%.4e' % x)
# import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.text import Text
import matplotlib.tri as tri
import matplotlib
matplotlib.use('Agg')

def beta_func(t,t1,bmax):
    if t < t1:
        return bmax/t1*t 
    else:
        return bmax 
    
def load_logs(
        task, 
        arch='fdon1', 
        num_branch=4, 
        num_trunk=2, 
        width=32, 
        modes=12,
        num_train=600, 
        train_grid_size=129,
        mode='E',
        exp_root='./exps'):

    # average RL2 
    train_logs = {}
    test_logs = {}
    cpu_times = []
    gpu_times = []

    for seed in [0,1,2,3,4]:
        if arch == 'fno':
            base_name = (f"{task}_fno_nb{num_branch}_w{width}"
                        f"_m{modes}_res{train_grid_size}"
                        f"_ntrain{num_train}_seed{seed}")
        elif arch == 'fdon1':
            base_name = (f"{task}_fdon1_nb{num_branch}_nt{num_trunk}"
                        f"_w{width}_m{modes}_res{train_grid_size}"
                        f"_ntrain{num_train}_seed{seed}")
        else:  # fdon2
            base_name = (f"{task}_fdon2_nb{num_branch}_w{width}"
                        f"_m{modes}_res{train_grid_size}"
                        f"_ntrain{num_train}_seed{seed}")

        train_log = np.load(f"{exp_root}/{base_name}/{mode}/train_log.npy")
        test_log = np.load(f"{exp_root}/{base_name}/{mode}/test_log.npy")
        time_log = np.load(f"{exp_root}/{base_name}/{mode}/inference_times.npy", allow_pickle=True).item()
        nparams = np.load(f"{exp_root}/{base_name}/{mode}/params.npy", allow_pickle=True).item()
        cpu_times.append(time_log['cpu_time'])
        gpu_times.append(time_log['gpu_time'])
        
        train_logs[seed] = train_log 
        test_logs[seed] = test_log
    
    train_logs = pd.DataFrame(train_logs).values
    test_logs = pd.DataFrame(test_logs).values  
    return train_logs, test_logs, np.array(cpu_times).mean(), np.array(gpu_times).mean(), nparams

def training_dynamic_plot(output_file="./figs/training_dynamics.pdf"):
    # Create figure with subfigures (3 rows, 3 columns)
    f = plt.figure(figsize=(13.5, 9))
    subfigs = f.subfigures(3, 3).flatten()  # 9 subfigures

    # Create subplots
    axes = [subfig.subplots() for subfig in subfigs]

    n = np.arange(100)

    # Helper function to plot subfigures
    def plot_subplot(ax, n, train_logs, test_logs, title, label, color='r', model_name='FNO', is_dual=False, train_logs2=None, test_logs2=None):
        ax.plot(n, train_logs.mean(axis=1), f'-{color}', label=f'{model_name} train')
        ax.plot(n, test_logs.mean(axis=1), f'--{color}', label=f'{model_name} test')
        if is_dual:
            ax.plot(n, train_logs2.mean(axis=1), '-b', label='Type-2 Fourier-DON train')
            ax.plot(n, test_logs2.mean(axis=1), '--b', label='Type-2 Fourier-DON test')
        ax.set_ylim([5e-3, 1e-1])
        ax.set_yscale('log')
        ax.set_xlabel('number of iterations')
        ax.set_ylabel('relative $L_2$ error')
        ax.set_title(title)
        ax.legend(loc='upper right', prop={'size': 10})
        ax.grid(True, which="both", ls="--", alpha=0.5)
        return Text(-0.01, 0.96, label, fontsize=12, ha="left", va="top")

    # Plot 1: (Single-temperature) Z -> E
    train_logs, test_logs, _, _, _ = load_logs("heat-1T-zsquares", arch='fno')
    label_a = plot_subplot(axes[0], n, train_logs, test_logs, '(Single-temperature) $\\mathbf{Z} \\rightarrow \\mathbf{E}$', '(a)')
    subfigs[0].add_artist(label_a)

    # Plot 2: (Single-temperature) Z x t1 -> E
    train_logs, test_logs, _, _, _ = load_logs("heat-1T-zsquares-t1", arch='fdon1')
    train_logs2, test_logs2, _, _, _ = load_logs("heat-1T-zsquares-t1", arch='fdon2')
    label_b = plot_subplot(axes[1], n, train_logs, test_logs, '(Single-temperature) $\\mathbf{Z} \\times t_1 \\rightarrow \\mathbf{E}$', '(b)', model_name='Type-1 Fourier-DON', is_dual=True, train_logs2=train_logs2, test_logs2=test_logs2)
    subfigs[1].add_artist(label_b)

    # Plot 3: (Single-temperature) Z x t1 x beta_max -> E
    train_logs, test_logs, _, _, _ = load_logs("heat-1T-zsquares-t1-bmax", arch='fdon1')
    train_logs2, test_logs2, _, _, _ = load_logs("heat-1T-zsquares-t1-bmax", arch='fdon2')
    label_c = plot_subplot(axes[2], n, train_logs, test_logs, '(Single-temperature) $\\mathbf{Z} \\times t_1 \\times b_{\\text{max}} \\rightarrow \\mathbf{E}$', '(c)', model_name='Type-1 Fourier-DON', is_dual=True, train_logs2=train_logs2, test_logs2=test_logs2)
    subfigs[2].add_artist(label_c)

    # Plot 4: (Two-temperature) Z -> E
    train_logs, test_logs, _, _, _ = load_logs("heat-2T-zsquares", mode='E', arch='fno', train_grid_size=257)
    label_d = plot_subplot(axes[3], n, train_logs, test_logs, '(Two-temperature) $\\mathbf{Z} \\rightarrow \\mathbf{E}$', '(d)')
    subfigs[3].add_artist(label_d)

    # Plot 5: (Two-temperature) Z x t1 -> E
    train_logs, test_logs, _, _, _ = load_logs("heat-2T-zsquares-t1", mode='E', arch='fdon1', train_grid_size=257)
    train_logs2, test_logs2, _, _, _ = load_logs("heat-2T-zsquares-t1", mode='E', arch='fdon2', train_grid_size=257)
    label_e = plot_subplot(axes[4], n, train_logs, test_logs, '(Two-temperature) $\\mathbf{Z} \\times t_1 \\rightarrow \\mathbf{E}$', '(e)', model_name='Type-1 Fourier-DON', is_dual=True, train_logs2=train_logs2, test_logs2=test_logs2)
    subfigs[4].add_artist(label_e)

    # Plot 6: (Two-temperature) Z x t1 x beta_max -> E
    train_logs, test_logs, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='E', arch='fdon1', train_grid_size=257)
    train_logs2, test_logs2, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='E', arch='fdon2', train_grid_size=257)
    label_f = plot_subplot(axes[5], n, train_logs, test_logs, '(Two-temperature) $\\mathbf{Z} \\times t_1 \\times b_{\\text{max}} \\rightarrow \\mathbf{E}$', '(f)', model_name='Type-1 Fourier-DON', is_dual=True, train_logs2=train_logs2, test_logs2=test_logs2)
    subfigs[5].add_artist(label_f)

    # Plot 7: (Two-temperature) Z -> T
    train_logs, test_logs, _, _, _ = load_logs("heat-2T-zsquares", mode='T', arch='fno', train_grid_size=257)
    label_g = plot_subplot(axes[6], n, train_logs, test_logs, '(Two-temperature) $\\mathbf{Z} \\rightarrow \\mathbf{T}$', '(g)')
    subfigs[6].add_artist(label_g)

    # Plot 8: (Two-temperature) Z x t1 -> T
    train_logs, test_logs, _, _, _ = load_logs("heat-2T-zsquares-t1", mode='T', arch='fdon1', train_grid_size=257)
    train_logs2, test_logs2, _, _, _ = load_logs("heat-2T-zsquares-t1", mode='T', arch='fdon2', train_grid_size=257)
    label_h = plot_subplot(axes[7], n, train_logs, test_logs, '(Two-temperature) $\\mathbf{Z} \\times t_1 \\rightarrow \\mathbf{T}$', '(h)', model_name='Type-1 Fourier-DON', is_dual=True, train_logs2=train_logs2, test_logs2=test_logs2)
    subfigs[7].add_artist(label_h)

    # Plot 9: (Two-temperature) Z x t1 x beta_max -> T
    train_logs, test_logs, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='T', arch='fdon1', train_grid_size=257)
    train_logs2, test_logs2, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='T', arch='fdon2', train_grid_size=257)
    label_i = plot_subplot(axes[8], n, train_logs, test_logs, '(Two-temperature) $\\mathbf{Z} \\times t_1 \\times b_{\\text{max}} \\rightarrow \\mathbf{T}$', '(i)', model_name='Type-1 Fourier-DON', is_dual=True, train_logs2=train_logs2, test_logs2=test_logs2)
    subfigs[8].add_artist(label_i)

    # Save and close
    # plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(f)

def heat_2T_sample_plot(idx=0, task='zsquares-t1-bmax', output_file="./figs/heat_2T_preds.pdf"):
    # Validate task
    valid_tasks = ['zsquares', 'zsquares-t1', 'zsquares-t1-bmax']
    if task not in valid_tasks:
        raise ValueError(f"Task must be one of {valid_tasks}")

    # Load datasets
    def load_mat_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return scipy.io.loadmat(path)

    datasets = {
        task: load_mat_file(f'../dataset/nd/heat-2T-{task}_r257.mat')
    }
    print('(log)  sucessfully loaded the reference files')

    grid = datasets[task]['grid']
    if not (len(grid[0].shape) == 2 and len(grid[1].shape) == 2):
        raise ValueError("Grid must contain 2D arrays")

    # Load reference solutions for E and T
    refs = {}
    for mode in ['E', 'T']:
        data = datasets[task][f'sol_{mode}']
        if idx >= len(data) - 100:
            raise ValueError(f"Index {idx} out of range for {task} dataset with {len(data)} samples")
        refs[f'{task}_{mode}'] = data[-100+idx]

    # Load predictions for E and T
    pred_paths = {}
    for mode in ['E', 'T']:
        if task == 'zsquares':
            # FNO for Type-1, Type-2 unavailable
            pred_paths[f'type1_{task}_{mode}'] = f"./exps/heat-2T-{task}_fno_nb4_w32_m12_res257_ntrain600_seed0/{mode}/pred.npy"
            pred_paths[f'type2_{task}_{mode}'] = None  # Type-2 not available for zsquares
        else:
            # Fourier-DON for Type-1 and Type-2
            pred_paths[f'type1_{task}_{mode}'] = f"./exps/heat-2T-{task}_fdon1_nb4_nt2_w32_m12_res257_ntrain600_seed0/{mode}/pred.npy"
            pred_paths[f'type2_{task}_{mode}'] = f"./exps/heat-2T-{task}_fdon2_nb4_w32_m12_res257_ntrain600_seed0/{mode}/pred.npy"

    preds = {}
    for key, path in pred_paths.items():
        if path is None:
            warnings.warn(f"Type-2 predictions not available for task '{task}'")
            preds[key] = None
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Prediction file not found: {path}")
            preds[key] = np.load(path)[idx]
            print(f'(log)  sucessfully loaded the prediction file: {key} {preds[key].shape}')
    print('(log)  sucessfully loaded the prediction files')

    # Create figure
    f = plt.figure(figsize=(10, 10))
    subfigs = f.subfigures(4, 3).flatten()
    axes = [subfig.subplots() for subfig in subfigs]

    # Hide empty subplots (original d and k)
    axes[3].set_visible(False)
    axes[9].set_visible(False)

    # Helper function for plotting
    def plot_pcolormesh(ax, subfig, grid, data, title, label, vmin, vmax, cmap='jet'):
        pcm = ax.pcolormesh(grid[0], grid[1], data, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.contour(grid[0], grid[1], data, colors='k', alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect('equal')
        subfig.colorbar(pcm, ax=ax)
        label = Text(-0.01, 0.96, label, fontsize=12, transform=subfig.transSubfigure, ha="left", va="top")
        subfig.add_artist(label)

    # Plot settings
    titles = {
        f'{task}_E': f"FEM $( \\mathbf{{E}})$",
        f'type1_{task}_E_pred': f"{'FNO' if task == 'zsquares' else 'Type-1 Fourier-DON'} $( \\tilde {{ \\mathbf{{E}} }} )$",
        f'type2_{task}_E_pred': f"Type-2 Fourier-DON $( \\tilde{{ \\mathbf{{E}} }} )$",
        f'{task}_T': f"FEM ($\\tilde{{ \\mathbf{{T}} }})$",
        f'type1_{task}_T_pred': f"{'FNO' if task == 'zsquares' else 'Type-1 Fourier-DON'} $( \\tilde{{ \\mathbf{{T}} }})$",
        f'type2_{task}_T_pred': f"Type-2 Fourier-DON $( \\tilde{{ \\mathbf{{T}} }} )$"
    }
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
    vmin_ref = {'E': 0, 'T': 0}
    vmax_ref = {'E': 10, 'T': 2}
    vmin_err = {'E': 0, 'T': 0}
    vmax_err = {'E': 0.5, 'T': 0.1}

    # Plot subplots in 4x3 layout
    # Row 1: FEM E (a), Type-1 E (b), Type-2 E (c)
    plot_pcolormesh(axes[0], subfigs[0], grid, refs[f'{task}_E'], titles[f'{task}_E'], labels[0], vmin_ref['E'], vmax_ref['E'])
    plot_pcolormesh(axes[1], subfigs[1], grid, preds[f'type1_{task}_E'], titles[f'type1_{task}_E_pred'], labels[1], vmin_ref['E'], vmax_ref['E'])
    if preds[f'type2_{task}_E'] is not None:
        plot_pcolormesh(axes[2], subfigs[2], grid, preds[f'type2_{task}_E'], titles[f'type2_{task}_E_pred'], labels[2], vmin_ref['E'], vmax_ref['E'])
    print('(log)  sucessfully plot Row 1: FEM E (a), Type-1 E (b), Type-2 E (c)')

    # Row 2: Hidden, Error Type-1 E (d), Error Type-2 E (e)
    error = np.abs(refs[f'{task}_E'] - preds[f'type1_{task}_E'])
    plot_pcolormesh(axes[4], subfigs[4], grid, error, f"Abs. Error $( | \\mathbf{{E}} - \\tilde {{ \\mathbf{{E}} }} | )$ ", labels[3], vmin_err['E'], vmax_err['E'])
    if preds[f'type2_{task}_E'] is not None:
        error = np.abs(refs[f'{task}_E'] - preds[f'type2_{task}_E'])
        plot_pcolormesh(axes[5], subfigs[5], grid, error, f"Abs. Error $( | \\mathbf{{E}} - \\tilde {{ \\mathbf{{E}} }} | )$ ", labels[4], vmin_err['E'], vmax_err['E'])
    print('(log)  sucessfully plot Row 2: Hidden, Error Type-1 E (d), Error Type-2 E (e)')

    # Row 3: FEM T (f), Type-1 T (g), Type-2 T (h)
    plot_pcolormesh(axes[6], subfigs[6], grid, refs[f'{task}_T'], titles[f'{task}_T'], labels[5], vmin_ref['T'], vmax_ref['T'])
    plot_pcolormesh(axes[7], subfigs[7], grid, preds[f'type1_{task}_T'], titles[f'type1_{task}_T_pred'], labels[6], vmin_ref['T'], vmax_ref['T'])
    if preds[f'type2_{task}_T'] is not None:
        plot_pcolormesh(axes[8], subfigs[8], grid, preds[f'type2_{task}_T'], titles[f'type2_{task}_T_pred'], labels[7], vmin_ref['T'], vmax_ref['T'])
    print('(log)  sucessfully plot Row 3: FEM T (f), Type-1 T (g), Type-2 T (h)')

    # Row 4: Hidden, Error Type-1 T (i), Error Type-2 T (j)
    error = np.abs(refs[f'{task}_T'] - preds[f'type1_{task}_T'])
    plot_pcolormesh(axes[10], subfigs[10], grid, error, f"Abs. Error $( | \\mathbf{{T}} - \\tilde {{ \\mathbf{{T}} }} | )$ ", labels[8], vmin_err['T'], vmax_err['T'])
    if preds[f'type2_{task}_T'] is not None:
        error = np.abs(refs[f'{task}_T'] - preds[f'type2_{task}_T'])
        plot_pcolormesh(axes[11], subfigs[11], grid, error, f"Abs. Error $( | \\mathbf{{T}} - \\tilde {{ \\mathbf{{T}} }} | )$ ", labels[9], vmin_err['T'], vmax_err['T'])
    print('(log)  sucessfully plot Row 4: Hidden, Error Type-1 T (i), Error Type-2 T (j)')

    # Save and close
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(f)

def heat_1T_seq_sample_plot(idx=0, output_file="./figs/heat_1T_seq_preds.pdf"):
    # Load datasets
    def load_mat_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return scipy.io.loadmat(path)

    datasets = {}
    for snapshot in ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0']:
        datasets[snapshot] = load_mat_file(f'../dataset/nd_seq/{snapshot}/heat-1T-zsquares-t1-bmax_r129.mat')
    grid = datasets['0.2']['grid']
    if not (len(grid[0].shape) == 2 and len(grid[1].shape) == 2):
        raise ValueError("Grid must contain 2D arrays")
    
    # Load reference solutions
    refs = {}
    for snapshot in ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0']:
        data = datasets[snapshot]['sol']
        if idx >= len(data) - 100:
            raise ValueError(f"Index {idx} out of range for {snapshot} dataset with {len(data)} samples")
        refs[f'{snapshot}'] = data[-100+idx]
        print(f'(log)  sucessfully loaded the reference files : {snapshot} {refs[snapshot].shape}')

    # Load predictions
    fdon1_preds = {}
    fdon2_preds = {}
    for snapshot in ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0']:
        pred_path = f"./seq_exps/{snapshot}/heat-1T-zsquares-t1-bmax_fdon1_nb4_nt2_w32_m12_res129_ntrain600_seed0/E/pred.npy"
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        fdon1_preds[snapshot] = np.load(pred_path)[idx]
        print(f'(log)  sucessfully loaded the fdon1 prediction file: {snapshot} {fdon1_preds[snapshot].shape}')
        
        pred_path = f"./seq_exps/{snapshot}/heat-1T-zsquares-t1-bmax_fdon2_nb4_w32_m12_res129_ntrain600_seed0/E/pred.npy"
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        fdon2_preds[snapshot] = np.load(pred_path)[idx]
        print(f'(log)  sucessfully loaded the fdon2 prediction file: {snapshot} {fdon2_preds[snapshot].shape}')
        
    # Create figure
    f = plt.figure(figsize=(15, 15))
    subfigs = f.subfigures(5, 5).flatten()
    axes = [subfig.subplots() for subfig in subfigs]

    def plot_pcolormesh(ax, subfig, grid, data, title, label, vmin, vmax, cmap='jet'):
        pcm = ax.pcolormesh(grid[0], grid[1], data, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.contour(grid[0], grid[1], data, colors='k', alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect('equal')
        subfig.colorbar(pcm, ax=ax)
        label = Text(-0.01, 0.96, label, fontsize=12, transform=subfig.transSubfigure, ha="left", va="top")
        subfig.add_artist(label)

    # Plot subplots in 5x5 layout
    # Row 1: FEM 0.4 (a), FEM 0.8 (b), FEM 1.2 (c) FEM 1.6 (d) FEM 2.0 (e)
    for i, snapshot in enumerate(['0.4', '0.8', '1.2', '1.6', '2.0']):
        if i == 0:
            label = '(a) FEM Reference'
        else:
            label = ''
        plot_pcolormesh(axes[i], subfigs[i], grid, refs[snapshot], f"${{t={snapshot}}}$", label, 0, 10)
    print('(log)  sucessfully plot Row 1: FEM 0.4 (a), FEM 0.8 (b), FEM 1.2 (c) FEM 1.6 (d) FEM 2.0 (e)')

    # Row 2: Type-1 0.4 (f), Type-1 0.8 (g), Type-1 1.2 (h) Type-1 1.6 (i) Type-1 2.0 (j)
    for i, snapshot in enumerate(['0.4', '0.8', '1.2', '1.6', '2.0']):
        if i == 0:
            label = '(b) Type-1 Fourier-DON Predictions'
        else:
            label = ''
        plot_pcolormesh(axes[i+5], subfigs[i+5], grid, fdon1_preds[snapshot], f"${{t={snapshot}}}$", label, 0, 10)
    print('(log)  sucessfully plot Row 2: Type-1 0.4 (f), Type-1 0.8 (g), Type-1 1.2 (h) Type-1 1.6 (i) Type-1 2.0 (j)')

    # Row 3: Type-1 err 0.4 (k), Type-1 err 0.8 (l), Type-1 err 1.2 (m) Type-1 err 1.6 (n) Type-1 err 2.0 (o)
    for i, snapshot in enumerate(['0.4', '0.8', '1.2', '1.6', '2.0']):
        if i == 0:
            label = '(c) Type-1 Fourier-DON Abs. Error'
        else:
            label = ''
        error = np.abs(refs[snapshot] - fdon1_preds[snapshot])
        plot_pcolormesh(axes[i+10], subfigs[i+10], grid, error, f"${{t={snapshot}}}$", label, 0, 0.5)
    print('(log)  sucessfully plot Row 3: Type-1 Err 0.4 (f), Type-1 Err 0.8 (g), Type-1 Err 1.2 (h) Type-1 Err 1.6 (i) Type-1 Err 2.0 (j)')

    # Row 4: Type-2 0.4 (p), Type-2 0.8 (q), Type-2 1.2 (r) Type-2 1.6 (s) Type-2 2.0 (t)
    for i, snapshot in enumerate(['0.4', '0.8', '1.2', '1.6', '2.0']):
        if i == 0:
            label = '(d) Type-2 Fourier-DON Predictions'
        else:
            label = ''
        plot_pcolormesh(axes[i+15], subfigs[i+15], grid, fdon2_preds[snapshot], f"${{t={snapshot}}}$", label, 0, 10)
    print('(log)  sucessfully plot Row 4: Type-2 0.4 (p), Type-2 0.8 (q), Type-2 1.2 (r) Type-2 1.6 (s) Type-2 2.0 (t)')

    # Row 5: Type-2 err 0.4 (u), Type-2 err 0.8 (v), Type-2 err 1.2 (w) Type-2 err 1.6 (x) Type-2 err 2.0 (y)
    for i, snapshot in enumerate(['0.4', '0.8', '1.2', '1.6', '2.0']):
        if i == 0:
            label = '(e) Type-2 Fourier-DON Abs. Error'
        else:
            label = ''
        error = np.abs(refs[snapshot] - fdon2_preds[snapshot])
        plot_pcolormesh(axes[i+20], subfigs[i+20], grid, error, f"${{t={snapshot}}}$", label, 0, 0.5)
    print('(log)  sucessfully plot Row 5: Type-2 err 0.4 (u), Type-2 err 0.8 (v), Type-2 err 1.2 (w) Type-2 err 1.6 (x) Type-2 err 2.0 (y)')
    # Save and close
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(f)

def ablation_study(output_file="./figs/ablation_study.pdf"):
    # Create figure with subfigures (2 rows, 3 columns)
    f = plt.figure(figsize=(12, 6))  # Adjusted height for 2 rows
    subfigs = f.subfigures(2, 4).flatten()  # 8 subfigures
    axes = [subfig.subplots() for subfig in subfigs]

    # Helper function to plot error bars
    def plot_errorbar_subplot(ax, x, err_E, err_T, x_label, title, label, ylim):
        err_E_yerr = [err_E[:, 0] - err_E[:, 1], err_E[:, 2] - err_E[:, 0]]
        err_T_yerr = [err_T[:, 0] - err_T[:, 1], err_T[:, 2] - err_T[:, 0]]
        ax.errorbar(x, err_E[:, 0], yerr=err_E_yerr, fmt='-o', label='$\\mathbf{E}$', markersize=2, capsize=4, c='blue')
        ax.errorbar(x, err_T[:, 0], yerr=err_T_yerr, fmt='-o', label='$\\mathbf{T}$', markersize=2, capsize=4, c='green')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Relative $L_2$ Error')
        ax.set_ylim(ylim)
        # Set specific x-axis ticks based on x_label
        if x_label == 'Number of training samples':
            ax.set_xticks([200, 400, 600])
        elif x_label == 'Number of channels of Fourier layers':
            ax.set_xticks([8, 16, 32])
        elif x_label == 'Number of Fourier layers':
            ax.set_xticks([2, 4, 6])
        elif x_label == 'Number of modes of Fourier layers':
            ax.set_xticks([8, 12, 16])
        ax.set_title(title)
        ax.legend(loc='upper right', prop={'size': 10})
        ax.grid(True)
        return Text(-0.01, 0.96, label, fontsize=12, ha="left", va="top")

    # Row 1: Type-1 Fourier-DON
    # Plot 1: Training samples (Type-1 Fourier-DON)
    ntrains = [200, 400, 600]
    err_E, err_T = [], []
    for n in ntrains:
        _, logs_E_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon1', 
            train_grid_size=257, mode='E',
            num_train=n)
        _, logs_T_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon1', 
            train_grid_size=257, mode='T',
            num_train=n)
        err_E.append([np.mean(logs_E_test[-1]), np.min(logs_E_test[-1]), np.max(logs_E_test[-1])])
        err_T.append([np.mean(logs_T_test[-1]), np.min(logs_T_test[-1]), np.max(logs_T_test[-1])])
    err_E, err_T = np.array(err_E), np.array(err_T)
    label_a = plot_errorbar_subplot(axes[0], ntrains, err_E, err_T, 'Number of training samples', 
                                    'Type-1 Fourier-DON', '(a)', (0, 5.5e-2))
    subfigs[0].add_artist(label_a)
    print('(log)  sucessfully plot Training samples (Type-1 Fourier-DON)')

    # Plot 2: Channels (Type-1 Fourier-DON)
    ws = [8, 16, 32]
    err_E, err_T = [], []
    for w in ws:
        _, logs_E_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon1',
            train_grid_size=257, mode='E',
            width=w)
        _, logs_T_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon1',
            train_grid_size=257, mode='T',
            width=w)
        err_E.append([np.mean(logs_E_test[-1]), np.min(logs_E_test[-1]), np.max(logs_E_test[-1])])
        err_T.append([np.mean(logs_T_test[-1]), np.min(logs_T_test[-1]), np.max(logs_T_test[-1])])
    err_E, err_T = np.array(err_E), np.array(err_T)
    label_b = plot_errorbar_subplot(axes[1], ws, err_E, err_T, 'Number of channels of Fourier layers', 
                                    'Type-1 Fourier-DON', '(b)', (0, 5.5e-2))
    subfigs[1].add_artist(label_b)
    print('(log)  sucessfully plot Channels (Type-1 Fourier-DON)')

    # Plot 3: Branch layers (Type-1 Fourier-DON)
    bs = [2, 4, 6]
    err_E, err_T = [], []
    for b in bs:
        _, logs_E_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon1',
            train_grid_size=257, mode='E',
            num_branch=b)
        _, logs_T_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon1',
            train_grid_size=257, mode='T',
            num_branch=b)
        err_E.append([np.mean(logs_E_test[-1]), np.min(logs_E_test[-1]), np.max(logs_E_test[-1])])
        err_T.append([np.mean(logs_T_test[-1]), np.min(logs_T_test[-1]), np.max(logs_T_test[-1])])
    err_E, err_T = np.array(err_E), np.array(err_T)
    label_c = plot_errorbar_subplot(axes[2], bs, err_E, err_T, 'Number of Fourier layers', 
                                    'Type-1 Fourier-DON', '(c)', (0, 5.5e-2))
    subfigs[2].add_artist(label_c)
    print('(log)  sucessfully plot Branch layers (Type-1 Fourier-DON)')

    # Plot 4: Branch layers (Type-1 Fourier-DON)
    ms = [8, 12, 16]
    err_E, err_T = [], []
    for m in ms:
        _, logs_E_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon1',
            train_grid_size=257, mode='E',
            modes=m)
        _, logs_T_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon1',
            train_grid_size=257, mode='T',
            modes=m)
        err_E.append([np.mean(logs_E_test[-1]), np.min(logs_E_test[-1]), np.max(logs_E_test[-1])])
        err_T.append([np.mean(logs_T_test[-1]), np.min(logs_T_test[-1]), np.max(logs_T_test[-1])])
    err_E, err_T = np.array(err_E), np.array(err_T)
    label_d = plot_errorbar_subplot(axes[3], ms, err_E, err_T, 'Number of modes of Fourier layers', 
                                    'Type-1 Fourier-DON', '(d)', (0, 5.5e-2))
    subfigs[3].add_artist(label_d)
    print('(log)  sucessfully plot Branch layers (Type-1 Fourier-DON)')


    # Row 2: Type-2 Fourier-DON
    # Plot 5: Training samples (Type-2 Fourier-DON)
    ntrains = [200, 400, 600]
    err_E, err_T = [], []
    for n in ntrains:
        _, logs_E_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon2', 
            train_grid_size=257, mode='E',
            num_train=n)
        _, logs_T_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon2', 
            train_grid_size=257, mode='T',
            num_train=n)
        err_E.append([np.mean(logs_E_test[-1]), np.min(logs_E_test[-1]), np.max(logs_E_test[-1])])
        err_T.append([np.mean(logs_T_test[-1]), np.min(logs_T_test[-1]), np.max(logs_T_test[-1])])
    err_E, err_T = np.array(err_E), np.array(err_T)
    label_e = plot_errorbar_subplot(axes[4], ntrains, err_E, err_T, 'Number of training samples', 
                                    'Type-2 Fourier-DON', '(e)', (0, 5.5e-2))
    subfigs[4].add_artist(label_e)
    print('(log)  sucessfully plot Branch layers (Type-1 Fourier-DON)')

    # Plot 6: Channels (Type-2 Fourier-DON)
    ws = [8, 16, 32]
    err_E, err_T = [], []
    for w in ws:
        _, logs_E_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon2',
            train_grid_size=257, mode='E',
            width=w)
        _, logs_T_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon2',
            train_grid_size=257, mode='T',
            width=w)
        err_E.append([np.mean(logs_E_test[-1]), np.min(logs_E_test[-1]), np.max(logs_E_test[-1])])
        err_T.append([np.mean(logs_T_test[-1]), np.min(logs_T_test[-1]), np.max(logs_T_test[-1])])
    err_E, err_T = np.array(err_E), np.array(err_T)
    label_f = plot_errorbar_subplot(axes[5], ws, err_E, err_T, 'Number of channels of Fourier layers', 
                                    'Type-2 Fourier-DON', '(f)', (0, 5.5e-2))
    subfigs[5].add_artist(label_f)
    print('(log)  sucessfully plot Channels (Type-2 Fourier-DON)')

    # Plot 7: Branch layers (Type-2 Fourier-DON)
    bs = [2, 4, 6]
    err_E, err_T = [], []
    for b in bs:
        _, logs_E_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon2',
            train_grid_size=257, mode='E',
            num_branch=b)
        _, logs_T_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon2',
            train_grid_size=257, mode='T',
            num_branch=b)
        err_E.append([np.mean(logs_E_test[-1]), np.min(logs_E_test[-1]), np.max(logs_E_test[-1])])
        err_T.append([np.mean(logs_T_test[-1]), np.min(logs_T_test[-1]), np.max(logs_T_test[-1])])
    err_E, err_T = np.array(err_E), np.array(err_T)
    label_g = plot_errorbar_subplot(axes[6], bs, err_E, err_T, 'Number of Fourier layers', 
                                    'Type-2 Fourier-DON', '(g)', (0, 5.5e-2))
    subfigs[6].add_artist(label_g)
    print('(log)  sucessfully plot Channels (Type-2 Fourier-DON)')

    # Plot 7: Modes layers (Type-2 Fourier-DON)
    ms = [8, 12, 16]
    err_E, err_T = [], []
    for m in ms:
        _, logs_E_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon2',
            train_grid_size=257, mode='E',
            modes=m)
        _, logs_T_test, _, _, _ = load_logs(
            task='heat-2T-zsquares-t1-bmax', arch='fdon2',
            train_grid_size=257, mode='T',
            modes=m)
        err_E.append([np.mean(logs_E_test[-1]), np.min(logs_E_test[-1]), np.max(logs_E_test[-1])])
        err_T.append([np.mean(logs_T_test[-1]), np.min(logs_T_test[-1]), np.max(logs_T_test[-1])])
    err_E, err_T = np.array(err_E), np.array(err_T)
    label_h = plot_errorbar_subplot(axes[7], ms, err_E, err_T, 'Number of modes of Fourier layers', 
                                    'Type-2 Fourier-DON', '(h)', (0, 5.5e-2))
    subfigs[7].add_artist(label_h)
    print('(log)  sucessfully plot Modes layers (Type-2 Fourier-DON)')

    # Save and close
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(f)

def print_table():
    # Collect results
    # Process heat-1T tasks
    # Table 1
    print('Table 1')
    print("Accuarcy comparison (fno) \\\\")
    print("                               & Type-1(E)  & Type-2(T)   \\\\")
    tasknm = '$Z -> E$'
    _, fno_logs, _, _, _ = load_logs('heat-1T-zsquares', arch='fno', mode='E')
    print("{:<30} & {:<10.3e} & {:<10} \\\\".format(tasknm, fno_logs[-1].mean(), '-'))

    tasknm = '$Z -> E, T$'
    _, fno_logs_E, _, _, _ = load_logs('heat-2T-zsquares', arch='fno', mode='E', train_grid_size=257)
    _, fno_logs_T, _, _, _ = load_logs('heat-2T-zsquares', arch='fno', mode='T', train_grid_size=257)
    print("{:<30} & {:<10.3e} & {:<10.3e} \\\\".format(tasknm, fno_logs_E[-1].mean(), fno_logs_T[-1].mean()))
    print()

    # Table 2
    print('Table 2')
    print("Accuarcy comparison (fdon1/fdon2) \\\\")
    print("                               & Type-1(E)  & Type-2(E)  & Type-1(T)  & Type-2(T)  \\\\")

    tasknm = '$Z x t1 -> E$'
    _, fdon1_logs, _, _, _ = load_logs('heat-1T-zsquares-t1', arch='fdon1', mode='E')
    _, fdon2_logs, _, _, _ = load_logs('heat-1T-zsquares-t1', arch='fdon2', mode='E')
    print("{:<30} & {:<10.3e} & {:<10.3e} & {:<10} & {:<10} \\\\".format(tasknm, fdon1_logs[-1].mean(), fdon2_logs[-1].mean(), '-', '-'))

    tasknm = '$Z x t1 x bmax -> E$'
    _, fdon1_logs, _, _, _ = load_logs('heat-1T-zsquares-t1-bmax', arch='fdon1', mode='E')
    _, fdon2_logs, _, _, _ = load_logs('heat-1T-zsquares-t1-bmax', arch='fdon2', mode='E')
    print("{:<30} & {:<10.3e} & {:<10.3e} & {:<10} & {:<10} \\\\".format(tasknm, fdon1_logs[-1].mean(), fdon2_logs[-1].mean(), '-', '-'))

    tasknm = '$Z x t1 -> E, T'
    _, fdon1_logs_E, _, _, _ = load_logs('heat-2T-zsquares-t1', arch='fdon1', mode='E', train_grid_size=257)
    _, fdon1_logs_T, _, _, _ = load_logs('heat-2T-zsquares-t1', arch='fdon1', mode='T', train_grid_size=257)
    _, fdon2_logs_E, _, _, _ = load_logs('heat-2T-zsquares-t1', arch='fdon2', mode='E', train_grid_size=257)
    _, fdon2_logs_T, _, _, _ = load_logs('heat-2T-zsquares-t1', arch='fdon2', mode='T', train_grid_size=257)
    print("{:<30} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        tasknm, fdon1_logs_E[-1].mean(), fdon2_logs_E[-1].mean(), fdon1_logs_T[-1].mean(), fdon2_logs_T[-1].mean()))

    tasknm = '$Z x t1 x bmax -> E, T$'
    _, fdon1_logs_E, _, _, _ = load_logs('heat-2T-zsquares-t1-bmax', arch='fdon1', mode='E', train_grid_size=257)
    _, fdon1_logs_T, _, _, _ = load_logs('heat-2T-zsquares-t1-bmax', arch='fdon1', mode='T', train_grid_size=257)
    _, fdon2_logs_E, _, _, _ = load_logs('heat-2T-zsquares-t1-bmax', arch='fdon2', mode='E', train_grid_size=257)
    _, fdon2_logs_T, _, _, _ = load_logs('heat-2T-zsquares-t1-bmax', arch='fdon2', mode='T', train_grid_size=257)
    print("{:<30} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        tasknm, fdon1_logs_E[-1].mean(), fdon2_logs_E[-1].mean(), fdon1_logs_T[-1].mean(), fdon2_logs_T[-1].mean()))
    print()

    # Table 3
    print('Table 3')
    print("Efficiency comparison (CPU/GPU) \\\\")
    print("Tasks                          & Arch       & Params     & CPU        & GPU        & CPU Imp    & GPU Imp  \\\\")

    fem_heat1T = 0.434
    fem_heat2T = 11.348

    tasknm = '$Z -> E$'
    _, _, fno_cpu, fno_gpu, fno_params = load_logs('heat-1T-zsquares', arch='fno', mode='E')
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        tasknm, 'fno', fno_params, fno_cpu, fno_gpu, fem_heat1T/fno_cpu, fem_heat1T/fno_gpu))


    tasknm = '$Z x t1 -> E$'
    _, _, fdon1_cpu, fdon1_gpu, fdon1_params = load_logs('heat-1T-zsquares-t1', arch='fdon1', mode='E')
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        tasknm, 'fdon1', fdon1_params, fdon1_cpu, fdon1_gpu, fem_heat1T/fdon1_cpu, fem_heat1T/fdon1_gpu))
    _, _, fdon2_cpu, fdon2_gpu, fdon2_params = load_logs('heat-1T-zsquares-t1', arch='fdon2', mode='E')
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        '', 'fdon2', fdon2_params, fdon2_cpu, fdon2_gpu, fem_heat1T/fdon2_cpu, fem_heat1T/fdon2_gpu))


    tasknm = '$Z x t1 x bmax -> E$'
    _, _, fdon1_cpu, fdon1_gpu, fdon1_params = load_logs('heat-1T-zsquares-t1-bmax', arch='fdon1', mode='E')
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        tasknm, 'fdon1', fdon1_params, fdon1_cpu, fdon1_gpu, fem_heat1T/fdon1_cpu, fem_heat1T/fdon1_gpu))
    _, _, fdon2_cpu, fdon2_gpu, fdon2_params = load_logs('heat-1T-zsquares-t1-bmax', arch='fdon2', mode='E')
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        '', 'fdon2', fdon2_params, fdon2_cpu, fdon2_gpu, fem_heat1T/fdon2_cpu, fem_heat1T/fdon2_gpu))


    tasknm = '$Z -> E, T$'
    _, _, fno_cpu_E, fno_gpu_E, fno_params_E = load_logs('heat-2T-zsquares', arch='fno', mode='E', train_grid_size=257)
    _, _, fno_cpu_T, fno_gpu_T, fno_params_T = load_logs('heat-2T-zsquares', arch='fno', mode='T', train_grid_size=257)
    fno_cpu = fno_cpu_E + fno_cpu_T
    fno_gpu = fno_gpu_E + fno_gpu_T
    fno_params = fno_params_E + fno_params_T
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        tasknm, 'fno', fno_params, fno_cpu, fno_gpu, fem_heat2T/fno_cpu, fem_heat2T/fno_gpu))

    tasknm = '$Z x t1 -> E, T$'
    _, _, fdon1_cpu_E, fdon1_gpu_E, fdon1_params_E = load_logs('heat-2T-zsquares-t1', arch='fdon1', mode='E', train_grid_size=257)
    _, _, fdon1_cpu_T, fdon1_gpu_T, fdon1_params_T = load_logs('heat-2T-zsquares-t1', arch='fdon1', mode='T', train_grid_size=257)
    fdon1_cpu = fdon1_cpu_E + fdon1_cpu_T
    fdon1_gpu = fdon1_gpu_E + fdon1_gpu_T
    fdon1_params = fdon1_params_E + fdon1_params_T
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        tasknm, 'fdon1', fdon1_params, fdon1_cpu, fdon1_gpu, fem_heat2T/fdon1_cpu, fem_heat2T/fdon1_gpu))
    _, _, fdon2_cpu_E, fdon2_gpu_E, fdon2_params_E = load_logs('heat-2T-zsquares-t1', arch='fdon2', mode='E', train_grid_size=257)
    _, _, fdon2_cpu_T, fdon2_gpu_T, fdon2_params_T = load_logs('heat-2T-zsquares-t1', arch='fdon2', mode='T', train_grid_size=257)
    fdon2_cpu = fdon2_cpu_E + fdon2_cpu_T
    fdon2_gpu = fdon2_gpu_E + fdon2_gpu_T
    fdon2_params = fdon2_params_E + fdon2_params_T
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        "", 'fdon2', fdon2_params, fdon2_cpu, fdon2_gpu, fem_heat2T/fdon2_cpu, fem_heat2T/fdon2_gpu))

    tasknm = '$Z x t1 x bmax -> E, T$'
    _, _, fdon1_cpu_E, fdon1_gpu_E, fdon1_params_E = load_logs('heat-2T-zsquares-t1-bmax', arch='fdon1', mode='E', train_grid_size=257)
    _, _, fdon1_cpu_T, fdon1_gpu_T, fdon1_params_T = load_logs('heat-2T-zsquares-t1-bmax', arch='fdon1', mode='T', train_grid_size=257)
    fdon1_cpu = fdon1_cpu_E + fdon1_cpu_T
    fdon1_gpu = fdon1_gpu_E + fdon1_gpu_T
    fdon1_params = fdon1_params_E + fdon1_params_T
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        tasknm, 'fdon1', fdon1_params, fdon1_cpu, fdon1_gpu, fem_heat2T/fdon1_cpu, fem_heat2T/fdon1_gpu))
    _, _, fdon2_cpu_E, fdon2_gpu_E, fdon2_params_E = load_logs('heat-2T-zsquares-t1-bmax', arch='fdon2', mode='E', train_grid_size=257)
    _, _, fdon2_cpu_T, fdon2_gpu_T, fdon2_params_T = load_logs('heat-2T-zsquares-t1-bmax', arch='fdon2', mode='T', train_grid_size=257)
    fdon2_cpu = fdon2_cpu_E + fdon2_cpu_T
    fdon2_gpu = fdon2_gpu_E + fdon2_gpu_T
    fdon2_params = fdon2_params_E + fdon2_params_T
    print("{:<30} & {:<10} & {:<10d} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\".format(
        "", 'fdon2', fdon2_params, fdon2_cpu, fdon2_gpu, fem_heat2T/fdon2_cpu, fem_heat2T/fdon2_gpu))
    print()

    # Table 4
    print("Super-resolution results for Z x t1 x bmax -> E, T")
    print("Resolution           & Type-1(E)  & Type-2(E)  & Type-1(T)  & Type-2(T)  \\\\")
    _, test_logs_fdon1_E_65, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='E', arch='fdon1', train_grid_size=65)
    _, test_logs_fdon1_T_65, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='T', arch='fdon1', train_grid_size=65)
    _, test_logs_fdon2_E_65, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='E', arch='fdon2', train_grid_size=65)
    _, test_logs_fdon2_T_65, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='T', arch='fdon2', train_grid_size=65)
    test_logs_fdon1_E_65 = test_logs_fdon1_E_65[-1].mean()
    test_logs_fdon1_T_65 = test_logs_fdon1_T_65[-1].mean()
    test_logs_fdon2_E_65 = test_logs_fdon2_E_65[-1].mean()
    test_logs_fdon2_T_65 = test_logs_fdon2_T_65[-1].mean()

    print("{:<20} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\ ".format(
        "65x65", test_logs_fdon1_E_65, test_logs_fdon2_E_65, test_logs_fdon1_T_65, test_logs_fdon2_T_65))

    _, test_logs_fdon1_E_129, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='E', arch='fdon1', train_grid_size=129)
    _, test_logs_fdon1_T_129, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='T', arch='fdon1', train_grid_size=129)
    _, test_logs_fdon2_E_129, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='E', arch='fdon2', train_grid_size=129)
    _, test_logs_fdon2_T_129, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='T', arch='fdon2', train_grid_size=129)

    test_logs_fdon1_E_129 = test_logs_fdon1_E_129[-1].mean()
    test_logs_fdon1_T_129 = test_logs_fdon1_T_129[-1].mean()
    test_logs_fdon2_E_129 = test_logs_fdon2_E_129[-1].mean()
    test_logs_fdon2_T_129 = test_logs_fdon2_T_129[-1].mean()
    print("{:<20} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\ ".format(
        "129x129", test_logs_fdon1_E_129, test_logs_fdon2_E_129, test_logs_fdon1_T_129, test_logs_fdon2_T_129))

    _, test_logs_fdon1_T_257, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='T', arch='fdon1', train_grid_size=257)
    _, test_logs_fdon1_E_257, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='E', arch='fdon1', train_grid_size=257)
    _, test_logs_fdon2_E_257, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='E', arch='fdon2', train_grid_size=257)
    _, test_logs_fdon2_T_257, _, _, _ = load_logs("heat-2T-zsquares-t1-bmax", mode='T', arch='fdon2', train_grid_size=257)

    test_logs_fdon1_E_257 = test_logs_fdon1_E_257[-1].mean()
    test_logs_fdon1_T_257 = test_logs_fdon1_T_257[-1].mean()
    test_logs_fdon2_E_257 = test_logs_fdon2_E_257[-1].mean()
    test_logs_fdon2_T_257 = test_logs_fdon2_T_257[-1].mean()    
    print("{:<20} & {:<10.3e} & {:<10.3e} & {:<10.3e} & {:<10.3e} \\\\ ".format(
        "257x257", test_logs_fdon1_E_257, test_logs_fdon2_E_257, test_logs_fdon1_T_257, test_logs_fdon2_T_257))

    # Table 4
    print('Table 4')
    print("Snapshot time \\\\")
    fdon2_test_errs = []
    fdon1_test_errs = []
    for snap in ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0']:
        _, test_logs, _, _, _ = load_logs(
            "heat-1T-zsquares-t1-bmax", mode='E', arch='fdon2', 
            train_grid_size=129, exp_root="./seq_exps/{}".format(snap))
        fdon2_test_err = test_logs[-1].mean()

        _, test_logs, _, _, _ = load_logs(
            "heat-1T-zsquares-t1-bmax", mode='E', arch='fdon1', 
            train_grid_size=129, exp_root="./seq_exps/{}".format(snap))
        fdon1_test_err = test_logs[-1].mean()

        fdon2_test_errs.append(fdon2_test_err)
        fdon1_test_errs.append(fdon1_test_err) 
    
    format_string = "{:<20} & " + " & ".join(["{:<10.1f}"] * len(fdon2_test_errs)) + " \\\\"
    format_args = ['Snapshot Time'] + [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    print(format_string.format(*format_args))

    format_string = "{:<20} & " + " & ".join(["{:<10.3e}"] * len(fdon2_test_errs)) + " \\\\"
    format_args = ['fdon1'] + fdon1_test_errs 
    print(format_string.format(*format_args))

    format_args = ['fdon2'] + fdon2_test_errs
    print(format_string.format(*format_args))


if __name__ == '__main__':
    print("SAVE PLOTS")
    print("(log)  save sample results figure for task heat2T-zsquares-t1-bmax")
    heat_2T_sample_plot(idx=5)

    print("(log)  save ablation study figure for task heat2T-zsquares-t1-bmax")
    ablation_study()

    print("(log)  save training dynamics figure")
    training_dynamic_plot()

    print("(log) save sequence prediction figure")
    heat_1T_seq_sample_plot(idx=5)

    print("PRINT TABLES")
    print_table()

