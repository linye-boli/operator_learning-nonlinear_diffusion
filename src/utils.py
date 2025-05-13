import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import os 
import operator
from functools import reduce
from functools import partial

#################################################
# Utilities
#################################################

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MatReader(object):
    """A class to read MATLAB (.mat) files using scipy.io or h5py."""
    
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        """Initialize the MatReader with file path and conversion options.
        
        Args:
            file_path (str): Path to the .mat file
            to_torch (bool): Convert data to PyTorch tensors if True
            to_cuda (bool): Move data to CUDA device if True
            to_float (bool): Convert data to float32 if True
        """
        super(MatReader, self).__init__()
        
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        """Load the .mat file using appropriate library based on format."""
        try:
            self.data = scipy.io.loadmat(self.file_path)  # Try loading with scipy
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)  # Fall back to h5py for HDF5 format
            self.old_mat = False

    def load_file(self, file_path):
        """Load a new .mat file.
        
        Args:
            file_path (str): Path to the new .mat file
        """
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        """Read a specific field from the loaded .mat file.
        
        Args:
            field (str): Name of the field to read
            
        Returns:
            data: Processed data (numpy array or PyTorch tensor)
        """
        x = self.data[field]

        if not self.old_mat:  # Handle HDF5 format
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        """Set whether to move data to CUDA.
        
        Args:
            to_cuda (bool): New CUDA setting
        """
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        """Set whether to convert to PyTorch tensors.
        
        Args:
            to_torch (bool): New PyTorch conversion setting
        """
        self.to_torch = to_torch

    def set_float(self, to_float):
        """Set whether to convert to float32.
        
        Args:
            to_float (bool): New float conversion setting
        """
        self.to_float = to_float

class LpLoss(object):
    """Loss function computing relative/absolute Lp loss."""
    
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        """Initialize the LpLoss function.
        
        Args:
            d (int): Dimension of the data
            p (int): Order of the norm
            size_average (bool): Average the loss over batch if True
            reduction (bool): Apply reduction to output if True
        """
        super(LpLoss, self).__init__()
        
        assert d > 0 and p > 0  # Ensure valid dimensions and norm order
        
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        """Compute absolute Lp loss.
        
        Args:
            x (torch.Tensor): Predicted values
            y (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Absolute Lp loss
        """
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)  # Assume uniform mesh
        
        all_norms = (h**(self.d/self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        """Compute relative Lp loss.
        
        Args:
            x (torch.Tensor): Predicted values
            y (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Relative Lp loss
        """
        num_examples = x.size()[0]
        
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        """Compute the loss (calls rel by default).
        
        Args:
            x (torch.Tensor): Predicted values
            y (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Computed loss
        """
        return self.rel(x, y)

def count_params(model):
    """Count the total number of parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                   list(p.size() + (2,) if p.is_complex() else p.size()))
    return c

# Function to downsample a tensor by the specified factor
def downsample(tensor, ratio):
    if ratio == 1:
        return tensor
    # Apply strided slicing to spatial dimensions
    return tensor[..., ::ratio, ::ratio]

def load_data(args):
    """Load and preprocess training and test data based on task.
    
    Args:
        args (Namespace): Configuration arguments including task, data_root, etc.
        
    Returns:
        tuple: (train_loader, test_loader)
            - train_loader (dict): Dictionary of training DataLoaders
            - test_loader (dict): Dictionary of test DataLoaders
    """
    # Construct data file path
    data_path = os.path.join(args.data_root, f'{args.task}_r{args.grid_size}.mat')
    reader = MatReader(data_path)
    
    # Load data based on task type
    if args.task == 'heat-1T-zsquares':
        Z = (reader.read_field('coeff').unsqueeze(1) > 5).float() # Input coefficients
        E = reader.read_field('sol')  # Solution field
        
        assert Z.shape[0] >= (args.num_train + args.num_test), "Not enough samples: will cause train-test overlap"
        
        # Split into train and test sets
        Z_train = Z[:args.num_train]
        E_train = E[:args.num_train]
        Z_test = Z[-args.num_test:]
        E_test = E[-args.num_test:]

        # Downsample training data only
        Z_train = downsample(Z_train, args.ratio)
        E_train = downsample(E_train, args.ratio)

        # Create data loaders
        train_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, E_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, E_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader = {'E': train_loader_E}
        test_loader = {'E': test_loader_E} 

    elif args.task == 'heat-1T-zsquares-t1':
        Z = (reader.read_field('coeff').unsqueeze(1) > 5).float()
        E = reader.read_field('sol')
        xi = reader.read_field('t1')  # Additional parameter
        
        assert Z.shape[0] >= (args.num_train + args.num_test), "Not enough samples: will cause train-test overlap"
        
        Z_train = Z[:args.num_train]
        E_train = E[:args.num_train]
        xi_train = xi[:args.num_train]
        Z_test = Z[-args.num_test:]
        E_test = E[-args.num_test:]
        xi_test = xi[-args.num_test:]

        # Downsample training data only
        Z_train = downsample(Z_train, args.ratio)
        E_train = downsample(E_train, args.ratio)

        train_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, xi_train, E_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, xi_test, E_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader = {'E': train_loader_E}
        test_loader = {'E': test_loader_E} 

    elif args.task == 'heat-1T-zsquares-t1-bmax':
        Z = (reader.read_field('coeff').unsqueeze(1) > 5).float()
        E = reader.read_field('sol')
        t1 = reader.read_field('t1')
        bmax = reader.read_field('bmax')
        xi = torch.cat((t1, bmax), dim=1)  # Concatenate parameters
        
        assert Z.shape[0] >= (args.num_train + args.num_test), "Not enough samples: will cause train-test overlap"
        
        Z_train = Z[:args.num_train]
        E_train = E[:args.num_train]
        xi_train = xi[:args.num_train]
        Z_test = Z[-args.num_test:]
        E_test = E[-args.num_test:]
        xi_test = xi[-args.num_test:]

        # Downsample training data only
        Z_train = downsample(Z_train, args.ratio)
        E_train = downsample(E_train, args.ratio)

        train_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, xi_train, E_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, xi_test, E_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader = {'E': train_loader_E}
        test_loader = {'E': test_loader_E} 

    elif args.task == 'heat-2T-zsquares':
        Z = (reader.read_field('coeff').unsqueeze(1) > 5).float()
        E = reader.read_field('sol_E')  # E solution field
        T = reader.read_field('sol_T')  # T solution field
        
        assert Z.shape[0] >= (args.num_train + args.num_test), "Not enough samples: will cause train-test overlap"
        
        Z_train = Z[:args.num_train]
        E_train = E[:args.num_train]
        T_train = T[:args.num_train]
        Z_test = Z[-args.num_test:]
        E_test = E[-args.num_test:]
        T_test = T[-args.num_test:]

        # Downsample training data only
        Z_train = downsample(Z_train, args.ratio)
        E_train = downsample(E_train, args.ratio)
        T_train = downsample(T_train, args.ratio)

        train_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, E_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, E_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader_T = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, T_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_T = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, T_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader = {'E': train_loader_E, 'T': train_loader_T}
        test_loader = {'E': test_loader_E, 'T': test_loader_T}        

    elif args.task == 'heat-2T-zsquares-t1':
        Z = (reader.read_field('coeff').unsqueeze(1) > 5).float()
        E = reader.read_field('sol_E')
        T = reader.read_field('sol_T')
        xi = reader.read_field('t1')
        
        assert Z.shape[0] >= (args.num_train + args.num_test), "Not enough samples: will cause train-test overlap"
        
        Z_train = Z[:args.num_train]
        E_train = E[:args.num_train]
        T_train = T[:args.num_train]
        xi_train = xi[:args.num_train]
        Z_test = Z[-args.num_test:]
        E_test = E[-args.num_test:]
        T_test = T[-args.num_test:]
        xi_test = xi[-args.num_test:]

        # Downsample training data only
        Z_train = downsample(Z_train, args.ratio)
        E_train = downsample(E_train, args.ratio)
        T_train = downsample(T_train, args.ratio)

        train_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, xi_train, E_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, xi_test, E_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader_T = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, xi_train, T_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_T = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, xi_test, T_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader = {'E': train_loader_E, 'T': train_loader_T}
        test_loader = {'E': test_loader_E, 'T': test_loader_T}        

    elif args.task == 'heat-2T-zsquares-t1-bmax':
        Z = (reader.read_field('coeff').unsqueeze(1) > 5).float()
        E = reader.read_field('sol_E')
        T = reader.read_field('sol_T')
        t1 = reader.read_field('t1')
        bmax = reader.read_field('bmax')
        xi = torch.cat((t1, bmax), dim=1)  # Concatenate parameters
        
        assert Z.shape[0] >= (args.num_train + args.num_test), "Not enough samples: will cause train-test overlap"
        
        Z_train = Z[:args.num_train]
        E_train = E[:args.num_train]
        T_train = T[:args.num_train]
        xi_train = xi[:args.num_train]
        Z_test = Z[-args.num_test:]
        E_test = E[-args.num_test:]
        T_test = T[-args.num_test:]
        xi_test = xi[-args.num_test:]

        # Downsample training data only
        Z_train = downsample(Z_train, args.ratio)
        E_train = downsample(E_train, args.ratio)
        T_train = downsample(T_train, args.ratio)
        

        train_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, xi_train, E_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_E = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, xi_test, E_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader_T = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, xi_train, T_train), 
            batch_size=args.batch_size, shuffle=True
        )
        test_loader_T = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_test, xi_test, T_test), 
            batch_size=args.batch_size, shuffle=False
        )
        train_loader = {'E': train_loader_E, 'T': train_loader_T}
        test_loader = {'E': test_loader_E, 'T': test_loader_T}        

    # Modified print statement with detailed sample distribution
    print(f"(Log)   total {Z.shape[0]} samples loaded, first {args.num_train} samples used for training at {args.train_grid_size}x{args.train_grid_size}, "
          f" last {args.num_test} samples used for testing at {args.grid_size}x{args.grid_size}")
    
    return train_loader, test_loader

def save_model(model, output_dir):
    """Save the trained model's state_dict to a file.

    Args:
        model (torch.nn.Module): The trained PyTorch model (e.g., FNO2d, FDON2d, FDON2d_II).
        component_dir (str): Directory to save the model checkpoint (e.g., output_dir/component).

    Returns:
        str: Path to the saved model checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"(Log)   Saving Model to: {output_dir}")
 
    # Define the checkpoint path
    checkpoint_path = os.path.join(output_dir, 'model.pth')
    
    # Save the model's state_dict
    torch.save(model.state_dict(), checkpoint_path)
    print(f"(Log)   Model saved to {checkpoint_path}")
    
    return checkpoint_path

def save_dynamics(train_losses, test_losses, output_dir):
    """Save training dynamics to numpy files.
    
    Args:
        train_losses (np.array): Training losses per epoch
        test_losses (np.array): Test losses per epoch
        inference_times (dict): Dictionary of inference times
        predictions (np.array): Model predictions
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    print(f"(Log)   Saving Training Dynamics to: {output_dir}")
    
    # Save results as numpy files
    np.save(f"{output_dir}/train_log.npy", train_losses)
    np.save(f"{output_dir}/test_log.npy", test_losses)
    print(f"(Log)   Training Dynamics saved successfully in {output_dir}")

def save_predictions(predictions, output_dir):
    """Save predictions results to numpy files.
    
    Args:
        predictions (np.array): Model predictions
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    print(f"(Log)   Saving predictions to: {output_dir}")
    
    # Save results as numpy files
    np.save(f"{output_dir}/pred.npy", predictions)
    print(f"(Log)   Predictions saved successfully in {output_dir}")

def save_infertimes(inference_times, output_dir):
    """Save inference times to numpy files.
    
    Args:
        inference_times (dict): Dictionary of inference times
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    print(f"(Log)   Saving inference times to: {output_dir}")
    
    # Save results as numpy files
    np.save(f"{output_dir}/inference_times.npy", inference_times)
    print(f"(Log)   Inference times saved successfully in {output_dir}")    

def save_params(params, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    print(f"(Log)   Saving number of params to: {output_dir}")
    
    # Save results as numpy files
    np.save(f"{output_dir}/params.npy", params)
    print(f"(Log)   Number of params saved successfully in {output_dir}")    

def log_config(args):
    """Print the experiment configuration for the current run.

    Note: Trunk layers are only printed for fdon1 architecture (FDON2d), as fdon2 (FDON2d_II) and fno (FNO2d) do not use them.
    """
    print("\n=== Experiment Configuration ===")
    print(f"Task: {args.task}")
    print(f"Architecture: {args.arch}")
    print(f"Data Root: {args.data_root}")
    print(f"Training Samples: {args.num_train}")
    print(f"Test Samples: {args.num_test}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Fourier Modes: {args.modes}")
    print(f"Network Width: {args.width}")
    print(f"Random Seed: {args.seed}")
    print(f"Grid Size: {args.grid_size}")
    print(f"Branch Layers: {args.num_branch}")
    if args.arch == 'fdon1':
        print(f"Trunk Layers: {args.num_trunk}")
    if args.xi_dim > 0:
        print(f"Xi Dimension: {args.xi_dim}")
    print(f"Output Directory: {args.output_dir}")
    print("===============================\n")

def get_output_dir(args, task):
    """Generate the output directory path for a given task.

    Args:
        args (Namespace): Training configuration parameters.
        task (str): Task name.

    Returns:
        str: Path to the output directory (e.g., heat-1T-zsquares_fno_nb4_w32_m12_res129_ntrain600_seed2).
    """
    if args.arch == 'fno':
        base_name = (f"{task}_fno_nb{args.num_branch}_w{args.width}"
                     f"_m{args.modes}_res{args.train_grid_size}"
                     f"_ntrain{args.num_train}_seed{args.seed}")
    elif args.arch == 'fdon1':
        base_name = (f"{task}_fdon1_nb{args.num_branch}_nt{args.num_trunk}"
                     f"_w{args.width}_m{args.modes}_res{args.train_grid_size}"
                     f"_ntrain{args.num_train}_seed{args.seed}")
    else:  # fdon2
        base_name = (f"{task}_fdon2_nb{args.num_branch}_w{args.width}"
                     f"_m{args.modes}_res{args.train_grid_size}"
                     f"_ntrain{args.num_train}_seed{args.seed}")
    return os.path.join(args.output_dir, base_name)

def check_if_trained(output_dir, component):
    """Check if the experiment for a given task and component has already been trained.

    Args:
        output_dir (str): Base output directory for the experiment.
        component (str): Component name (e.g., 'E', 'T').

    Returns:
        bool: True if the experiment has already been trained (results exist), False otherwise.
    """
    result_file = os.path.join(output_dir, component, "model.pth")
    if os.path.exists(result_file):
        print(f"(Log)   Experiment already trained for {component} component in {output_dir}. Skipping training.")
        return True
    return False