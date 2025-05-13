import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# Fourier Layer
################################################################
class FourierConv2d(nn.Module):
    """2D Fourier convolution layer: FFT -> Linear Transform -> Inverse FFT."""
    
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        """Initialize the Fourier convolution layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            modes_x (int): Number of Fourier modes in x-direction
            modes_y (int): Number of Fourier modes in y-direction
        """
        super(FourierConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x  # Number of Fourier modes in x-direction
        self.modes_y = modes_y  # Number of Fourier modes in y-direction

        # Scaling factor for weight initialization to stabilize training
        scale = 1 / (in_channels * out_channels)
        # Complex weights for low and high frequency modes
        self.weights_low = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))
        self.weights_high = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    def complex_multiply(self, input_tensor, weights):
        """Perform complex multiplication between input and weights.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch, in_ch, x, y)
            weights (torch.Tensor): Weight tensor of shape (in_ch, out_ch, x, y)
            
        Returns:
            torch.Tensor: Result of shape (batch, out_ch, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x):
        """Forward pass of the Fourier convolution layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_ch, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_ch, height, width)
        """
        batch_size = x.shape[0]
        # Compute 2D FFT of the input
        x_fft = torch.fft.rfft2(x)

        # Initialize output Fourier coefficients with zeros
        height, width = x.size(-2), x.size(-1) // 2 + 1
        out_fft = torch.zeros(batch_size, self.out_channels, height, width, 
                            dtype=torch.cfloat, device=x.device)

        # Multiply low and high frequency modes with corresponding weights
        out_fft[:, :, :self.modes_x, :self.modes_y] = self.complex_multiply(
            x_fft[:, :, :self.modes_x, :self.modes_y], self.weights_low
        )
        out_fft[:, :, -self.modes_x:, :self.modes_y] = self.complex_multiply(
            x_fft[:, :, -self.modes_x:, :self.modes_y], self.weights_high
        )

        # Inverse FFT to return to spatial domain
        return torch.fft.irfft2(out_fft, s=(height, x.size(-1)))

################################################################
# Multi-Layer Perceptron (MLP) implemented by conv 1x1
################################################################
class MLP(nn.Module):
    """Simple MLP with 1x1 convolutions for channel-wise transformations."""
    
    def __init__(self, in_channels, out_channels, hidden_channels):
        """Initialize the MLP with 1x1 convolutions.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            hidden_channels (int): Number of hidden channels
        """
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_ch, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_ch, height, width)
        """
        x = F.gelu(self.conv1(x))  # Apply GELU activation after first convolution
        return self.conv2(x)  # Second convolution without activation

################################################################
# Multi-Layer Perceptron (MLP) implemented by linear layers
################################################################
class FCN(nn.Module):
    """Fully Connected Network (FCN) with linear layers for channel-wise transformations."""
    
    def __init__(self, in_channels, out_channels, hidden_channels):
        """Initialize the FCN with linear layers.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            hidden_channels (int): Number of hidden channels
        """
        super(FCN, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        """Forward pass of the FCN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_ch)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_ch)
        """
        x = F.gelu(self.linear1(x))  # Apply GELU activation after first linear layer
        return self.linear2(x)  # Second linear layer without activation

################################################################
# Fourier Neural Operator (FNO) Model
################################################################
class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D PDEs with configurable number of Fourier layers."""
    
    def __init__(self, modes_x, modes_y, width, num_layers=4):
        """Initialize the FNO2d model with a variable number of layers.
        
        Args:
            modes_x (int): Number of Fourier modes in x-direction
            modes_y (int): Number of Fourier modes in y-direction
            width (int): Number of channels (width of the network)
            num_layers (int, optional): Number of Fourier layers. Defaults to 4
        """
        super(FNO2d, self).__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.num_layers = num_layers
        self.padding = 9  # Padding for non-periodic inputs to avoid boundary effects

        # Input projection: (coeff, x, y) -> width channels
        self.input_proj = nn.Conv2d(3, width, kernel_size=1)
        
        # Dynamically create Fourier convolution layers, MLPs, and pointwise convolutions
        self.fourier_layers = nn.ModuleList([
            FourierConv2d(width, width, modes_x, modes_y) for _ in range(num_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            MLP(width, width, width) for _ in range(num_layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)
        ])

        # Output projection: width -> 1 channel
        self.output_proj = MLP(width, 1, width)
        
    def get_grid(self, shape, device):
        """Generate a 2D grid of normalized coordinates.
        
        Args:
            shape (tuple): Shape of the input tensor (batch, channels, height, width)
            device (torch.device): Device to create the grid on
            
        Returns:
            torch.Tensor: Grid tensor of shape (batch, 2, height, width)
        """
        batch_size, size_x, size_y = shape[0], shape[2], shape[3]
        # Generate normalized x and y coordinates
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float).reshape(1, 1, size_x, 1).repeat(batch_size, 1, 1, size_y)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float).reshape(1, 1, 1, size_y).repeat(batch_size, 1, size_x, 1)
        return torch.cat((grid_x, grid_y), dim=1).to(device)

    def forward(self, x):
        """Forward pass of the FNO2d model with variable layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, 1, height, width)
        """
        # Add grid coordinates to input
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        # Project input to wider channel space and pad
        x = self.input_proj(x)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        # Iterate through the specified number of Fourier blocks with skip connections
        for i in range(self.num_layers - 1):
            x = F.gelu(self.fourier_layers[i](x) + self.mlp_layers[i](self.conv_layers[i](x)))
        # Last layer without GELU activation
        x = self.fourier_layers[-1](x) + self.mlp_layers[-1](self.conv_layers[-1](x))

        # Remove padding and project to output space
        x = x[..., :-self.padding, :-self.padding]
        x = self.output_proj(x)
        return x

################################################################
# Fourier Deep Operator Network (FDON) Model
################################################################
class FDON2d(nn.Module):
    """Fourier Deep Operator Network for 2D PDEs with configurable branch and trunk layers."""
    
    def __init__(self, xi_dim, modes_x, modes_y, width, num_branch_layers=4, num_trunk_layers=2):
        """Initialize the FDON2d model with variable branch and trunk layers.
        
        Args:
            xi_dim (int): Dimension of the additional parameter input (xi)
            modes_x (int): Number of Fourier modes in x-direction
            modes_y (int): Number of Fourier modes in y-direction
            width (int): Number of channels (width of the network)
            num_branch_layers (int, optional): Number of Fourier layers in branch network. Defaults to 4
            num_trunk_layers (int, optional): Number of linear layers in trunk network. Defaults to 2
        """
        super(FDON2d, self).__init__()
        
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.num_branch_layers = num_branch_layers
        self.num_trunk_layers = num_trunk_layers
        self.padding = 9  # Padding for non-periodic inputs to avoid boundary effects

        # Branch network (FNO-like) setup
        # Input projection: (coeff, x, y) -> width channels
        self.input_proj = nn.Conv2d(3, width, kernel_size=1)        
        # Dynamically create Fourier convolution layers, MLPs, and pointwise convolutions
        self.fourier_layers = nn.ModuleList([
            FourierConv2d(width, width, modes_x, modes_y) for _ in range(num_branch_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            MLP(width, width, width) for _ in range(num_branch_layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(num_branch_layers)
        ])

        # Trunk network for processing xi
        self.trunk_layers = nn.ModuleList()
        # First layer takes xi_dim as input
        self.trunk_layers.append(FCN(xi_dim, width, width))
        # Subsequent layers take width as input
        for _ in range(num_trunk_layers - 1):
            self.trunk_layers.append(FCN(width, width, width))

    def get_grid(self, shape, device):
        """Generate a 2D grid of normalized coordinates.
        
        Args:
            shape (tuple): Shape of the input tensor (batch, channels, height, width)
            device (torch.device): Device to create the grid on
            
        Returns:
            torch.Tensor: Grid tensor of shape (batch, 2, height, width)
        """
        batch_size, size_x, size_y = shape[0], shape[2], shape[3]
        # Generate normalized x and y coordinates
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float).reshape(1, 1, size_x, 1).repeat(batch_size, 1, 1, size_y)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float).reshape(1, 1, 1, size_y).repeat(batch_size, 1, size_x, 1)
        return torch.cat((grid_x, grid_y), dim=1).to(device)

    def branch_forward(self, x):
        """Forward pass of the branch network (FNO component) with variable layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, width, height+padding, width+padding)
        """
        # Add grid coordinates to input
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        # Project input to wider channel space and pad
        x = self.input_proj(x)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        # Iterate through the specified number of Fourier blocks with skip connections
        for i in range(self.num_branch_layers - 1):
            x = F.gelu(self.fourier_layers[i](x) + self.mlp_layers[i](self.conv_layers[i](x)))
        # Last layer without GELU activation
        x = self.fourier_layers[-1](x) + self.mlp_layers[-1](self.conv_layers[-1](x))
        return x 
    
    def trunk_forward(self, x):
        """Forward pass of the trunk network (xi processing) with variable layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, xi_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, width)
        """
        # Iterate through the specified number of trunk layers
        for i in range(self.num_trunk_layers - 1):
            x = F.gelu(self.trunk_layers[i](x))
        # Last layer without GELU activation
        x = self.trunk_layers[-1](x)
        return x

    def forward(self, z, xi):
        """Forward pass of the FDON2d model combining branch and trunk outputs.
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch, 1, height, width)
            xi (torch.Tensor): Parameter tensor of shape (batch, xi_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, height, width)
        """
        x_branch = self.branch_forward(z)  # Process spatial input through branch network
        x_trunk = self.trunk_forward(xi)   # Process parameter input through trunk network
        # Combine branch and trunk outputs using einsum
        x = torch.einsum('bcxy,bc->bxy', x_branch, x_trunk)
        # Remove padding from the output
        x = x[..., :-self.padding, :-self.padding]
        return x
    
################################################################
# Fourier-DeepONet Model
################################################################
class FDON2d_II(nn.Module):
    """Fourier Deep Operator Network for 2D PDEs with configurable branch and trunk layers."""
    
    def __init__(self, xi_dim, modes_x, modes_y, width, num_layers=4):
        """Initialize the FDON2d model with variable branch and trunk layers.
        
        Args:
            xi_dim (int): Dimension of the additional parameter input (xi)
            modes_x (int): Number of Fourier modes in x-direction
            modes_y (int): Number of Fourier modes in y-direction
            width (int): Number of channels (width of the network)
            num_layers (int, optional): Number of Fourier layers. Defaults to 4
        """
        super(FDON2d_II, self).__init__()
        
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.num_layers = num_layers
        self.padding = 9  # Padding for non-periodic inputs to avoid boundary effects

        # Branch network (FNO-like) setup
        # Input projection: (coeff, x, y) -> width channels
        self.branch_encoder = nn.Conv2d(3, width, kernel_size=1)        
        self.trunk_encoder = nn.Linear(xi_dim, width)

        # Dynamically create Fourier convolution layers, MLPs, and pointwise convolutions
        self.fourier_layers = nn.ModuleList([
            FourierConv2d(width, width, modes_x, modes_y) for _ in range(num_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            MLP(width, width, width) for _ in range(num_layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)
        ])

        # Output projection: width -> 1 channel
        self.output_proj = MLP(width, 1, width)

    def get_grid(self, shape, device):
        """Generate a 2D grid of normalized coordinates.
        
        Args:
            shape (tuple): Shape of the input tensor (batch, channels, height, width)
            device (torch.device): Device to create the grid on
            
        Returns:
            torch.Tensor: Grid tensor of shape (batch, 2, height, width)
        """
        batch_size, size_x, size_y = shape[0], shape[2], shape[3]
        # Generate normalized x and y coordinates
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float).reshape(1, 1, size_x, 1).repeat(batch_size, 1, 1, size_y)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float).reshape(1, 1, 1, size_y).repeat(batch_size, 1, size_x, 1)
        return torch.cat((grid_x, grid_y), dim=1).to(device)

    def forward(self, z, xi):
        """Forward pass of the FDON2d model combining branch and trunk outputs.
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch, 1, height, width)
            xi (torch.Tensor): Parameter tensor of shape (batch, xi_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, height, width)
        """

        # Add grid coordinates to input
        grid = self.get_grid(z.shape, z.device)
        z = torch.cat((z, grid), dim=1)
        x_branch = self.branch_encoder(z)  # Process spatial input through branch network
        x_branch = F.pad(x_branch, [0, self.padding, 0, self.padding])
        x_trunk = self.trunk_encoder(xi)   # Process parameter input through trunk network

        # Combine branch and trunk outputs using einsum
        x = torch.einsum('bcxy,bc->bcxy', x_branch, x_trunk)

        # Iterate through the specified number of Fourier blocks with skip connections
        for i in range(self.num_layers - 1):
            x = F.gelu(self.fourier_layers[i](x) + self.mlp_layers[i](self.conv_layers[i](x)))
        # Last layer without GELU activation
        x = self.fourier_layers[-1](x) + self.mlp_layers[-1](self.conv_layers[-1](x))

        # Remove padding and project to output space
        x = x[..., :-self.padding, :-self.padding]
        x = self.output_proj(x)

        return x
