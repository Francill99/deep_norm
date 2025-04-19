import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod
from tqdm.auto import tqdm

class CNNClassifier(nn.Module):
    def __init__(self, 
                 conv_channels, 
                 kernel_size, 
                 mlp_layers, 
                 pool_every=1, 
                 dropout=0.5, 
                 input_size=(3, 32, 32)):
        """
        Constructs a CNN classifier.
        
        Parameters:
          conv_channels (list of int): A list specifying the channel dimensions for the convolutional layers.
            For example, [3, 32, 64] means the input has 3 channels, then a conv layer maps 3 → 32 channels,
            and the next conv layer maps 32 → 64.
          kernel_size (int): The kernel (window) size for all convolutional layers (assumed odd to allow padding).
          mlp_layers (list of int): A list defining the fully connected (MLP) layers after flattening.
            For example, [512, 128, num_classes].
          pool_every (int): Insert a MaxPool2d layer (with kernel size 2, stride 2) after every 'pool_every'
            convolutional layers.
          dropout (float): Dropout probability applied after each fully-connected layer (except the last).
          input_size (tuple): Input dimensions as (channels, height, width).
        """
        super(CNNClassifier, self).__init__()
        
        # --- Build the convolutional block ---
        conv_layers = []
        in_channels = conv_channels[0]
        for i, out_channels in enumerate(conv_channels[1:]):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                         padding=kernel_size // 2))
            conv_layers.append(nn.ReLU(inplace=True))
            # Insert pooling layer every 'pool_every' conv layer(s)
            if (i + 1) % pool_every == 0:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)
        
        # Determine the number of features after the conv block using a dummy forward pass.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            feat = self.features(dummy_input)
            self.flatten_dim = feat.view(1, -1).size(1)
        
        # --- Build the MLP (classifier) block ---
        mlp_layers_list = []
        prev_dim = self.flatten_dim
        # Add hidden layers with Linear, ReLU and dropout.
        for h in mlp_layers[:-1]:
            mlp_layers_list.append(nn.Linear(prev_dim, h))
            mlp_layers_list.append(nn.ReLU(inplace=True))
            if dropout > 0:
                mlp_layers_list.append(nn.Dropout(dropout))
            prev_dim = h
        # Final layer (without activation) for classification.
        mlp_layers_list.append(nn.Linear(prev_dim, mlp_layers[-1]))
        self.classifier = nn.Sequential(*mlp_layers_list)
        
    def forward(self, x):
        # Pass through convolutional features.
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        # Pass through classifier (MLP).
        x = self.classifier(x)
        return x

    def compute_model_norm(self):
        """
        Computes the model's spectral complexity, which is defined as:
        
        \[
        R_{F_A} = \Biggl(\prod_{i=1}^L \|A_i\|_\sigma\Biggr)
                      \Biggl(\sum_{i=1}^L \frac{\|A_{>i}\|_{2,1}^{\frac{2}{3}}}{\|A_i\|_\sigma^{\frac{2}{3}}}\Biggr)^{\frac{3}{2}},
        \]
        
        where the product is taken over all layers with weights (both convolutional and linear). For 
        each layer:
          - \(\|A_i\|_\sigma\) is its spectral norm, and 
          - \(\|A_{>i}\|_{2,1}\) is approximated by the sum of the (2,1)-norms of all subsequent layers,
            with the \((2,1)\)-norm defined as the sum of the \(\ell_2\) norms over columns.
        
        (Note: Here we assume the reference matrices \(M_i = 0\).)
        
        Returns:
          A scalar tensor representing the spectral complexity.
        """
        def spectral_norm(weight):
            # For convolutional weights, reshape to (out_channels, -1)
            if weight.ndim > 2:
                w = weight.view(weight.size(0), -1)
            else:
                w = weight
            s = torch.linalg.svdvals(w)
            return s[0]
        
        def two_one_norm(weight):
            if weight.ndim > 2:
                w = weight.view(weight.size(0), -1)
            else:
                w = weight
            # Sum of the \(\ell_2\) norms over columns.
            col_norms = torch.norm(w, p=2, dim=0)
            return torch.sum(col_norms)
        
        # Gather all modules with weights (Conv2d and Linear) from features and classifier.
        weighted_layers = []
        for module in list(self.features) + list(self.classifier):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weighted_layers.append(module)
        
        L = len(weighted_layers)
        if L == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        s_list = []
        t_list = []
        for module in weighted_layers:
            weight = module.weight
            s_list.append(spectral_norm(weight))
            t_list.append(two_one_norm(weight))
        
        prod_spec = torch.tensor(1.0, device=s_list[0].device)
        for s in s_list:
            prod_spec = prod_spec * s
        
        correction_sum = torch.tensor(0.0, device=prod_spec.device)
        for i in range(L):
            # For layer i, approximate \(\|A_{>i}\|_{2,1}\) by summing the (2,1)-norms of subsequent layers.
            if i < L - 1:
                t_sum = sum(t_list[i+1:])
            else:
                t_sum = torch.tensor(0.0, device=prod_spec.device)
            # Avoid division by zero.
            s_i = s_list[i] if s_list[i] > 0 else torch.tensor(1e-12, device=prod_spec.device)
            term = (t_sum ** (2.0/3.0)) / (s_i ** (2.0/3.0))
            correction_sum = correction_sum + term
        
        norm_value = prod_spec * (correction_sum ** (3.0/2.0))
        return norm_value

    def compute_margin_distribution(self, inputs, labels):
        """
        Computes the normalized margin distribution on a given batch.
        
        For a batch of \((x, y)\) pairs, the raw margin for each sample is computed as:
        
        \[
        m(x, y) = f(x)_y - \max_{j \neq y} f(x)_j,
        \]
        
        and the normalized margin is defined as:
        
        \[
        \tilde{m}(x, y) = \frac{m(x, y)}{R_{F_A}\,\Bigl(\frac{\|X\|_F}{n}\Bigr)},
        \]
        
        where:
          - \(R_{F_A}\) is the model norm computed with `compute_model_norm()`,
          - \(\|X\|_F\) is the Frobenius norm of the input batch (when each input is flattened),
          - \(n\) is the batch size.
          
        Parameters:
           inputs (Tensor): The input batch of shape \((n, C, H, W)\).
           labels (Tensor): The corresponding labels of shape \((n,)\).
        
        Returns:
           Tensor: A 1D tensor of normalized margin values for the batch.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)  # shape: (n, num_classes)
            batch_size = outputs.size(0)
            
            # Gather the scores for the correct classes.
            true_scores = outputs[torch.arange(batch_size), labels]
            # Set the scores for the true class to -infinity to compute the max of the remaining classes.
            outputs_clone = outputs.clone()
            outputs_clone[torch.arange(batch_size), labels] = -float('inf')
            max_incorrect = outputs_clone.max(dim=1)[0]
            
            raw_margins = true_scores - max_incorrect
            
            # Compute the model norm.
            model_norm = self.compute_model_norm()
            
            # Compute the Frobenius norm of the input batch.
            X_flat = inputs.view(batch_size, -1)
            X_norm = torch.norm(X_flat, p=2)
            
            # Denominator: R_{F_A} * (||X||_F / n)
            scaling_factor = model_norm * (X_norm / batch_size + 1e-12)
            
            normalized_margins = raw_margins / scaling_factor
        return normalized_margins