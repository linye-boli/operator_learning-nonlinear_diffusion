import os
import argparse
import torch
import numpy as np
from timeit import default_timer

# Import custom modules for neural networks and utility functions
from nets import FNO2d, FDON2d, FDON2d_II
from utils import LpLoss, count_params, load_data
from utils import log_config, get_output_dir, check_if_trained
from utils import save_model, save_predictions, save_infertimes, save_dynamics, save_params

################################################################
# Argument Parsing
################################################################
def parse_args():
    """Parse command-line arguments for configuring the training process.

    Computes xi_dim based on task (e.g., time or boundary conditions) and train_grid_size based on spatial sampling ratio.

    Returns:
        argparse.Namespace: Parsed arguments with computed attributes (xi_dim, train_grid_size).
    """
    parser = argparse.ArgumentParser(description="Training script for Fourier Neural Operator models")
    parser.add_argument('--data-root', type=str, default='../dataset/nd/',
                        help='Path to dataset directory')
    parser.add_argument('--task', type=str, default='heat-1T-zsquares',
                        help='Task name (e.g., heat-1T-zsquares, heat-2T-zsquares-t1)')
    parser.add_argument('--num-train', type=int, default=600,
                        help='Number of training samples')
    parser.add_argument('--num-test', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--modes', type=int, default=12,
                        help='Number of Fourier modes in x and y directions')
    parser.add_argument('--width', type=int, default=32,
                        help='Number of channels in the network')
    parser.add_argument('--grid-size', type=int, default=129,
                        help='Spatial grid resolution')
    parser.add_argument('--output-dir', type=str, default='../result',
                        help='Directory to save training results')
    parser.add_argument('--num-branch', type=int, default=4,
                        help='Number of branch layers (Fourier layers)')
    parser.add_argument('--num-trunk', type=int, default=2,
                        help='Number of trunk layers (linear layers for FDON2d)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--ratio', type=int, default=1,
                        help='Spatial sampling ratio (1, 2, or 4)')
    parser.add_argument('--arch', type=str, default='fno', choices=['fno', 'fdon1', 'fdon2'],
                        help='Model architecture: fno (FNO2d), fdon1 (FDON2d), fdon2 (FDON2d_II)')

    args = parser.parse_args()

    # Compute xi dimension based on task (e.g., t1 for time, bmax for boundary conditions)
    args.xi_dim = 0
    if 't1' in args.task:
        args.xi_dim += 1
    if 'bmax' in args.task:
        args.xi_dim += 1

    # Validate and compute training grid size based on sampling ratio
    if args.ratio not in [1, 2, 4]:
        raise ValueError(f"Spatial sampling ratio must be 1, 2, or 4, got {args.ratio}")
    if args.ratio == 1:
        args.train_grid_size = args.grid_size
    else:
        args.train_grid_size = args.grid_size // args.ratio + 1

    # Ensure task-architecture compatibility (FNO for basic tasks, FDON for tasks with xi)
    fdon_tasks = ['heat-1T-zsquares-t1', 'heat-1T-zsquares-t1-bmax',
                  'heat-2T-zsquares-t1', 'heat-2T-zsquares-t1-bmax']
    fno_tasks = ['heat-1T-zsquares', 'heat-2T-zsquares']
    if args.arch == 'fno' and args.task in fdon_tasks:
        raise ValueError(f"Architecture 'fno' is not compatible with fdon task: {args.task}")
    if args.arch in ['fdon1', 'fdon2'] and args.task in fno_tasks:
        raise ValueError(f"Architecture '{args.arch}' requires fdon task, got: {args.task}")

    return args

################################################################
# Training Functions
################################################################
def train_fno(model, train_loader, test_loader, args):
    """Train an FNO2d model using input z and evaluate on test data.

    Uses LpLoss for computing L2 loss and CosineAnnealingLR for learning rate scheduling.

    Args:
        model (FNO2d): The Fourier Neural Operator model.
        train_loader (DataLoader): Training data loader with input (z) and target (E_ref).
        test_loader (DataLoader): Test data loader with input (z) and target (E_ref).
        args (Namespace): Training configuration parameters.

    Returns:
        tuple: (model, train_losses, test_losses)
            - model (FNO2d): Trained model.
            - train_losses (np.array): Training losses per epoch.
            - test_losses (np.array): Test losses per epoch.
    """
    device = torch.device(f'cuda:{args.device}')
    model = model.to(device)
    print(f"(Train) Model initialized with {count_params(model)} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * (args.num_train // args.batch_size))
    loss_fn = LpLoss(size_average=False)  # L2 loss function for comparing predictions and targets

    train_losses, test_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        start_time = default_timer()
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # Reshape predictions to match target dimensions
            predictions = model(inputs).reshape(args.batch_size, args.train_grid_size, args.train_grid_size)
            loss = loss_fn(predictions.reshape(args.batch_size, -1), targets.reshape(args.batch_size, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs).reshape(args.batch_size, args.grid_size, args.grid_size)
                test_loss += loss_fn(predictions.reshape(args.batch_size, -1), targets.reshape(args.batch_size, -1)).item()

        train_loss /= args.num_train
        test_loss /= args.num_test

        print(f"(Train) Epoch {epoch+1}/{args.epochs}: Train L2 Loss = {train_loss:.4e}, "
              f"Test L2 Loss = {test_loss:.4e}, Time = {default_timer() - start_time:.2f}s")

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return model, np.array(train_losses), np.array(test_losses)

def train_fdon(model, train_loader, test_loader, args):
    """Train an FDON2d or FDON2d_II model using inputs z and xi, and evaluate on test data.

    Uses LpLoss for computing L2 loss and CosineAnnealingLR for learning rate scheduling.

    Args:
        model (FDON2d or FDON2d_II): The Fourier Deep Operator Network model.
        train_loader (DataLoader): Training data loader with input (z, xi) and target (E_ref).
        test_loader (DataLoader): Test data loader with input (z, xi) and target (E_ref).
        args (Namespace): Training configuration parameters.

    Returns:
        tuple: (model, train_losses, test_losses)
            - model (FDON2d or FDON2d_II): Trained model.
            - train_losses (np.array): Training losses per epoch.
            - test_losses (np.array): Test losses per epoch.
    """
    device = torch.device(f'cuda:{args.device}')
    model = model.to(device)
    print(f"(Train) Model initialized with {count_params(model)} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * (args.num_train // args.batch_size))
    loss_fn = LpLoss(size_average=False)  # L2 loss function for comparing predictions and targets

    train_losses, test_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        start_time = default_timer()
        train_loss = 0

        for inputs, xi, targets in train_loader:
            inputs, xi, targets = inputs.to(device), xi.to(device), targets.to(device)
            optimizer.zero_grad()
            # Reshape predictions to match target dimensions
            predictions = model(inputs, xi).reshape(args.batch_size, args.train_grid_size, args.train_grid_size)
            loss = loss_fn(predictions.reshape(args.batch_size, -1), targets.reshape(args.batch_size, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, xi, targets in test_loader:
                inputs, xi, targets = inputs.to(device), xi.to(device), targets.to(device)
                predictions = model(inputs, xi).reshape(args.batch_size, args.grid_size, args.grid_size)
                test_loss += loss_fn(predictions.reshape(args.batch_size, -1), targets.reshape(args.batch_size, -1)).item()

        train_loss /= args.num_train
        test_loss /= args.num_test

        print(f"(Train) Epoch {epoch+1}/{args.epochs}: Train L2 Loss = {train_loss:.4e}, "
              f"Test L2 Loss = {test_loss:.4e}, Time = {default_timer() - start_time:.2f}s")

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return model, np.array(train_losses), np.array(test_losses)

def infer_fno(model_checkpoint, test_loader, args, device='cuda'):
    """Perform inference using a trained FNO2d model loaded from a checkpoint.

    Computes predictions, test loss, and inference times on GPU (if available) and CPU.

    Args:
        model_checkpoint (str): Path to the trained model checkpoint (.pth file).
        test_loader (DataLoader): Test data loader with input (z) and target (E_ref).
        args (Namespace): Configuration parameters (e.g., modes, width, grid_size, num_test).
        device (str): Preferred device to run inference on ('cuda' or 'cpu').

    Returns:
        tuple: (predictions, test_rl2, inference_times)
            - predictions (np.array): Test set predictions.
            - test_rl2 (float): Relative L2 loss on the test set.
            - inference_times (dict): GPU and CPU inference times per sample.
    """
    from nets import FNO2d
    
    # Check GPU availability and set device
    use_gpu = torch.cuda.is_available() and device == 'cuda'
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    # Initialize FNO2d model with specified Fourier modes and width
    model = FNO2d(
        modes_x=args.modes, modes_y=args.modes,
        width=args.width, num_layers=args.num_branch
    )
    
    # Load model weights from checkpoint
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    params = count_params(model)
    print(f"(Infer) Loaded FNO2d model with {params} parameters for inference on {device}")

    # Compute predictions and test loss
    loss_fn = LpLoss(size_average=False)  # L2 loss function for evaluating predictions
    test_loss = 0
    predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).reshape(-1, args.grid_size, args.grid_size)
            test_loss += loss_fn(outputs.reshape(-1, args.grid_size * args.grid_size), 
                               targets.reshape(-1, args.grid_size * args.grid_size)).item()
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    test_rl2 = test_loss / args.num_test
    print(f"(Infer) Test Relative L2 Loss = {test_rl2:.4e}")

    # Measure inference times on GPU (if available) and CPU
    inference_times = {"gpu_time": None, "cpu_time": None}
    if use_gpu:
        # Measure GPU inference time
        start_time = default_timer()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                _ = model(inputs)
        inference_times["gpu_time"] = (default_timer() - start_time) / args.num_test
        print(f"(Infer) GPU average inference time per sample: {inference_times['gpu_time']:.4e} seconds")

    # Measure CPU inference time
    model = model.cpu()
    start_time = default_timer()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to('cpu')
            _ = model(inputs)
    inference_times["cpu_time"] = (default_timer() - start_time) / args.num_test
    print(f"(Infer) CPU average inference time per sample: {inference_times['cpu_time']:.4e} seconds")

    return predictions, test_rl2, inference_times

def infer_fdon(model_checkpoint, test_loader, args, device='cuda'):
    """Perform inference using a trained FDON2d or FDON2d_II model loaded from a checkpoint.

    Computes predictions, test loss, and inference times on GPU (if available) and CPU.

    Args:
        model_checkpoint (str): Path to the trained model checkpoint (.pth file).
        test_loader (DataLoader): Test data loader with input (z, xi) and target (E_ref).
        args (Namespace): Configuration parameters (e.g., xi_dim, modes, width, arch, num_test).
        device (str): Preferred device to run inference on ('cuda' or 'cpu').

    Returns:
        tuple: (predictions, test_rl2, inference_times)
            - predictions (np.array): Test set predictions.
            - test_rl2 (float): Relative L2 loss on the test set.
            - inference_times (dict): GPU and CPU inference times per sample.
    """
    from nets import FDON2d, FDON2d_II
    
    # Check GPU availability and set device
    use_gpu = torch.cuda.is_available() and device == 'cuda'
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    # Initialize model based on architecture
    if args.arch == 'fdon1':
        model = FDON2d(
            xi_dim=args.xi_dim, modes_x=args.modes, modes_y=args.modes,
            width=args.width, num_branch_layers=args.num_branch, num_trunk_layers=args.num_trunk
        )
    else:  # fdon2
        model = FDON2d_II(
            xi_dim=args.xi_dim, modes_x=args.modes, modes_y=args.modes,
            width=args.width, num_layers=args.num_branch
        )
    
    # Load model weights from checkpoint
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    params = count_params(model)
    print(f"(Infer) Loaded {args.arch.upper()} model with {params} parameters for inference on {device}")

    # Compute predictions and test loss
    loss_fn = LpLoss(size_average=False)  # L2 loss function for evaluating predictions
    test_loss = 0
    predictions = []
    with torch.no_grad():
        for inputs, xi, targets in test_loader:
            inputs, xi, targets = inputs.to(device), xi.to(device), targets.to(device)
            outputs = model(inputs, xi).reshape(-1, args.grid_size, args.grid_size)
            test_loss += loss_fn(outputs.reshape(-1, args.grid_size * args.grid_size), 
                               targets.reshape(-1, args.grid_size * args.grid_size)).item()
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    test_rl2 = test_loss / args.num_test
    print(f"(Infer) Test Relative L2 Loss = {test_rl2:.4e}")

    # Measure inference times on GPU (if available) and CPU
    inference_times = {"gpu_time": None, "cpu_time": None}
    if use_gpu:
        # Measure GPU inference time
        start_time = default_timer()
        with torch.no_grad():
            for inputs, xi, _ in test_loader:
                inputs, xi = inputs.to(device), xi.to(device)
                _ = model(inputs, xi)
        inference_times["gpu_time"] = (default_timer() - start_time) / args.num_test
        print(f"(Infer) GPU average inference time per sample: {inference_times['gpu_time']:.4e} seconds")

    # Measure CPU inference time
    model = model.cpu()
    start_time = default_timer()
    with torch.no_grad():
        for inputs, xi, _ in test_loader:
            inputs, xi = inputs.to('cpu'), xi.to('cpu')
            _ = model(inputs, xi)
    inference_times["cpu_time"] = (default_timer() - start_time) / args.num_test
    print(f"(Infer) CPU average inference time per sample: {inference_times['cpu_time']:.4e} seconds")

    return predictions, test_rl2, inference_times

################################################################
# Main Execution
################################################################
if __name__ == "__main__":
    """Main execution script for training and inference of FNO models.

    Parses arguments, loads data, trains models for each component, and performs inference.
    """
    # Parse arguments and set random seeds for reproducibility
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load training and test data (returns dict of DataLoaders for each component)
    train_loader, test_loader = load_data(args)

    # Log configuration to output directory
    log_config(args)

    # Validate task against supported tasks
    supported_tasks = [
        'heat-1T-zsquares', 'heat-2T-zsquares',
        'heat-1T-zsquares-t1', 'heat-1T-zsquares-t1-bmax',
        'heat-2T-zsquares-t1', 'heat-2T-zsquares-t1-bmax'
    ]
    if args.task not in supported_tasks:
        raise ValueError(f"Unsupported task: {args.task}. Supported tasks: {supported_tasks}")

    # Determine components (E for single output, E and T for multi-output tasks)
    components = ['E'] if args.task in ['heat-1T-zsquares', 'heat-1T-zsquares-t1', 'heat-1T-zsquares-t1-bmax'] else ['E', 'T']
    output_dir = get_output_dir(args, args.task)

    # Train model for each component
    for component in components:
        component_dir = os.path.join(output_dir, component)
        # Skip training if component is already trained
        if check_if_trained(output_dir, component):
            continue

        print(f"(Train) Starting training for {args.task} task - {component} component")
        if args.arch == 'fno':
            # Initialize FNO2d model with specified Fourier modes and layers
            model = FNO2d(
                modes_x=args.modes, modes_y=args.modes,
                width=args.width, num_layers=args.num_branch)
            model, train_losses, test_losses = train_fno(
                model, train_loader[component], test_loader[component], args)
        else:  # fdon1 or fdon2
            if args.arch == 'fdon1':
                # Initialize FDON2d model with xi input and trunk layers
                model = FDON2d(
                    xi_dim=args.xi_dim, modes_x=args.modes, modes_y=args.modes,
                    width=args.width, num_branch_layers=args.num_branch, num_trunk_layers=args.num_trunk)
            else:  # fdon2
                # Initialize FDON2d_II model with xi input
                model = FDON2d_II(
                    xi_dim=args.xi_dim, modes_x=args.modes, modes_y=args.modes,
                    width=args.width, num_layers=args.num_branch)
            model, train_losses, test_losses = train_fdon(
                model, train_loader[component], test_loader[component], args)

        # Save trained model weights to component directory
        save_model(model, component_dir)
        # Save training and test loss dynamics (e.g., as numpy arrays)
        save_dynamics(train_losses, test_losses, component_dir)
    
    # Perform inference for each component
    for component in components:
        component_dir = os.path.join(output_dir, component)
        model_checkpoint = os.path.join(component_dir, 'model.pth')
        print(f"(Infer) Starting inference for {args.task} task - {component} component")

        if args.arch == 'fno':
            # Initialize FNO2d model for inference
            model = FNO2d(
                modes_x=args.modes, modes_y=args.modes,
                width=args.width, num_layers=args.num_branch)
            predictions, test_rl2, inference_times = infer_fno(
                model_checkpoint, test_loader[component], args)            
        else:  # fdon1 or fdon2
            if args.arch == 'fdon1':
                # Initialize FDON2d model for inference
                model = FDON2d(
                    xi_dim=args.xi_dim, modes_x=args.modes, modes_y=args.modes,
                    width=args.width, num_branch_layers=args.num_branch, num_trunk_layers=args.num_trunk)
            else:  # fdon2
                # Initialize FDON2d_II model for inference
                model = FDON2d_II(
                    xi_dim=args.xi_dim, modes_x=args.modes, modes_y=args.modes,
                    width=args.width, num_layers=args.num_branch)
            predictions, test_rl2, inference_times = infer_fdon(
                model_checkpoint, test_loader[component], args)
        
        # Save predictions (e.g., as numpy arrays) to component directory
        save_predictions(predictions, component_dir)        
        # Save inference times (GPU and CPU) to component directory
        save_infertimes(inference_times, component_dir)
        # Save Number of Params (GPU and CPU) to component directory
        save_params(np.array(count_params(model)), component_dir)