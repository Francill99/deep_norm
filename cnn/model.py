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
                 input_size=(3, 32, 32),
                 initialization_factor: float = 1.0,
                ):
        super().__init__()
        self._init_factor = initialization_factor
        
        # --- Build the convolutional block ---
        conv_layers = []
        in_channels = conv_channels[0]
        for i, out_channels in enumerate(conv_channels[1:]):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, 
                                         kernel_size=kernel_size, 
                                         padding=kernel_size // 2))
            conv_layers.append(nn.ReLU(inplace=True))
            if (i + 1) % pool_every == 0:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)
        
        # figure out flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            feat = self.features(dummy)
            self.flatten_dim = feat.view(1, -1).size(1)
        
        # --- Build the MLP ---
        mlp = []
        prev = self.flatten_dim
        for h in mlp_layers[:-1]:
            mlp.append(nn.Linear(prev, h))
            mlp.append(nn.ReLU(inplace=True))
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            prev = h
        mlp.append(nn.Linear(prev, mlp_layers[-1]))
        self.classifier = nn.Sequential(*mlp)
        
        # --- NOW scale *all* conv/linear weights ---
        self._scale_initial_weights()
    
    def _scale_initial_weights(self):
        # 1.0 → no change; anything else scales
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # multiply the *existing* initialization
                m.weight.data.mul_(self._init_factor)
                # leave bias as whatever it was (PyTorch default = 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

    def compute_l1_norm(self):
        """
        Computes the entry-wise L1 norm: sum of absolute values of all weights.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            total += torch.sum(torch.abs(param))
        return total

    def compute_frobenius_norm(self):
        """
        Computes the Frobenius norm: sqrt(sum of squares of all weights).
        """
        total_sq = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            total_sq += torch.sum(param ** 2)
        return torch.sqrt(total_sq)

    def compute_group_2_1_norm(self):
        """
        Computes the (2,1) group norm: sum of L2 norms of columns per layer.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in list(self.features) + list(self.classifier):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                if weight.ndim > 2:
                    w = weight.view(weight.size(0), -1)
                else:
                    w = weight
                col_norms = torch.norm(w, p=2, dim=0)
                total += torch.sum(col_norms)
        return total

    def compute_spectral_norm(self):
        """
        Computes the product of spectral norms (largest singular value) across layers.
        """
        prod_spec = torch.tensor(1.0, device=next(self.parameters()).device)
        for module in list(self.features) + list(self.classifier):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                if weight.ndim > 2:
                    w = weight.view(weight.size(0), -1)
                else:
                    w = weight
                s = torch.linalg.svdvals(w)[0]
                prod_spec *= s
        return prod_spec

    def compute_path_norm(self):
        """
        Approximates the path norm via dynamic programming over channels:
        - For Conv2d: sum absolute weights over spatial dims to get channel connectivity.
        - For Linear: use absolute weight matrix directly.
        """
        device = next(self.parameters()).device
        prev = None
        for module in list(self.features) + list(self.classifier):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                if weight.ndim > 2:
                    # weight shape: [out_channels, in_channels, kH, kW]
                    abs_w = torch.sum(torch.abs(weight), dim=(2, 3))
                else:
                    abs_w = torch.abs(weight)
                # abs_w: [out_units, in_units]
                if prev is None:
                    prev = torch.ones(abs_w.size(1), device=device)
                curr = abs_w.matmul(prev)
                prev = curr
        return torch.sum(prev)

    def compute_fisher_rao_norm(self, inputs, labels):
        """
        Approximates the Fisher-Rao norm via diagonal Fisher information diag(F) ≈ E[grad^2].
        Requires one batch of (inputs, labels).
        """
        self.eval()
        # compute gradients of log-likelihood
        self.zero_grad()
        outputs = F.log_softmax(self(inputs), dim=1)
        batch_size = outputs.size(0)
        log_probs = outputs[torch.arange(batch_size), labels]
        loss = -log_probs.mean()
        loss.backward()
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            if param.grad is not None:
                # diag(F)_i ≈ (grad_i)^2 ; FR norm ≈ sum theta_i^2 * diag(F)_i
                total += torch.sum((param.detach() ** 2) * (param.grad.detach() ** 2))
        return torch.sqrt(total)


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
    
    # ---------- Dario's magical comments ----------

    # Explanation for why the Conv2d layer can be flattened

    # A convolutional layer applies a linear transformation that maps the input feature maps 
    # to output feature maps by performing a convolution operation. Although the weight tensor 
    # of a Conv2d layer is 4-dimensional (with shape [out_channels, in_channels, kernel_h, kernel_w]), 
    # we can 'flatten' it into a 2D matrix where each row corresponds to one output channel and 
    # each column represents the weights applied to a local patch of the input (unfolded into a vector).

    # This flattening is justified because the convolution operation can be equivalently formulated 
    # as a matrix multiplication: by using an "im2col" operation, the convolution is transformed into 
    # a multiplication between a large (but structured) matrix and the vectorized input. In both cases, 
    # the linear mapping defined by the convolution has the same singular values. Thus, evaluating 
    # the largest singular value of the flattened weight matrix provides the spectral norm of the 
    # convolutional layer.

    # Explanation for why the MaxPool layer is 1-Lipschitz:
    #
    # A non-overlapping max-pooling operation divides the input into disjoint patches and takes the maximum value
    # within each patch. For any two input patches, the difference between the maximum values is bounded by the 
    # largest difference between corresponding elements in the patches. This implies that, for each patch,
    # |max(patch1) - max(patch2)| ≤ max(|x_i - y_i|) for x_i and y_i in the respective patches.
    #
    # Since the patches are non-overlapping, this bound holds independently for each patch. When the differences
    # are aggregated (e.g., with the Euclidean norm across patches), the overall change in the pooled output is 
    # no larger than the change in the input. Therefore, the max-pooling layer does not amplify the input 
    # differences, making it 1-Lipschitz with respect to the l2 norm.

    # Explanation for why the ReLU layer is 1-Lipschitz:
    #
    # |ReLU(x) - ReLU(y)| = |max(x, 0) - max(y, 0)| ≤ |x - y|

    def compute_spectral_complexity_ALT(self):
        """
        Estimates the spectral complexity:
        
         R_A = (product of spectral norms)
               * (Sum_i [||A_i^T||_{2,1}^{2/3} / ||A_i||_{sigma}^{2/3} ])^{3/2}

        Here, we choose M_i = 0 for each layer.
        """
        # We track two things:
        # 1) The product of spectral norms (prod_sn).
        # 2) The sum over i of (||A_i^T||_{2,1} / ||A_i||_sigma^(2/3)).

        prod_sn = 1.0
        sum_ratio = 0.0

        def matrix_21_norm(weight_matrix):
            """
            Computes the (2,1)-group norm of a 2D matrix:
            sum of the Euclidean (L2) norms of each column.
            """
            # weight_matrix shape: [out_dim, in_dim]
            col_norms = weight_matrix.norm(dim=0)  # L2 norm of each column
            return col_norms.sum()

        def spectral_norm_via_svd(weight_matrix):
            """
            Computes the largest singular value using torch.linalg.svdvals.
            """
            with torch.no_grad():
                # Compute only the singular values (which are returned in descending order).
                S = torch.linalg.svdvals(weight_matrix)
            return S[0].item()  # Return the largest singular value.
        
        # ---- 1) Convolution layers ----
        for module in self.features:
            if isinstance(module, nn.Conv2d):
                # Flatten to 2D: [out_channels, in_channels * kernel_h * kernel_w]
                weight_2d = module.weight.data.view(module.out_channels, -1)
                
                sn = spectral_norm_via_svd(weight_2d)
                prod_sn *= sn

                # group (2,1) norm of A_i^T
                norm_21 = matrix_21_norm(weight_2d.T)
                
                # Add ratio term
                sum_ratio += float((norm_21 / (sn + 1e-12))**(2.0/3.0))

        # ---- 2) MLP (Linear) layers ----
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                weight_2d = module.weight.data  # shape [out_features, in_features]
                
                sn = spectral_norm_via_svd(weight_2d)
                prod_sn *= sn

                norm_21 = matrix_21_norm(weight_2d)
                if sn > 0:
                    sum_ratio += float((norm_21 / (sn + 1e-12))**(2.0/3.0))
        
        # Final combination
        spectral_complexity = prod_sn * (sum_ratio**(3.0/2.0))
        return spectral_complexity
    
    def compute_margin_distribution_ALT(self, inputs, labels):
        """
        Computes the normalized margin distribution on a given batch.
        
        raw margin = f(x)_y - max_{j != y} f(x)_j
        Normalized margin = (raw margin) / (R_{F_A} * (||X||_2 / n))

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
            model_complexity = self.compute_spectral_complexity_ALT
            
            # Compute the Frobenius norm of the input batch.
            X_flat = inputs.view(batch_size, -1)
            X_norm = torch.norm(X_flat, p=2)
            
            # Denominator: R_{F_A} * (||X||_F / n)
            scaling_factor = model_complexity * (X_norm / batch_size + 1e-12)
            
            normalized_margins = raw_margins / scaling_factor
            
        return normalized_margins