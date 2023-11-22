import torch
from typing import Optional, Tuple
from torch import Tensor, nn
from torchmdnet.models.utils import (
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
# Creates a skew-symmetric tensor from a vector
def vector_to_skewtensor(vector):
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zero,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


# Creates a symmetric traceless tensor from the outer product of a vector with itself
def vector_to_symtensor(vector):
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S


# Full tensor decomposition into irreducible components
def decompose_tensor(tensor):
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return I, A, S


# Computes Frobenius norm
def tensor_norm(tensor):
    return (tensor**2).sum((-2, -1))


def der_cutoff(distances, cutoff_upper):
    return -(math.pi / (2 * cutoff_upper)) * torch.sin(math.pi * distances / cutoff_upper)

def grad_rbfs(dist, betas, means, alpha, cutoff_upper, cutoff_fn, grad_output):
    
    dist_expanded = dist.unsqueeze(-1)  # Shape becomes [5, 1]
    exp_arg = alpha * (-dist_expanded)
    exp_term = torch.exp(exp_arg)
    smearing_term = torch.exp(-betas * (exp_term - means.unsqueeze(0)) ** 2)  # Applying broadcasting
    output = cutoff_fn(dist_expanded) * smearing_term  # Shape will be [5, 3]

     
    der_cutoff = -(math.pi / (2 * cutoff_upper)) * torch.sin(math.pi * dist_expanded / cutoff_upper)
    doutput_ddist_expanded = (smearing_term * der_cutoff + cutoff_fn(dist_expanded) * 
                          (-2 * betas * (exp_term - means.unsqueeze(0)) * smearing_term * 
                           exp_term * -alpha)) * grad_output
    doutput_ddist_expanded_summed = doutput_ddist_expanded.sum(dim=-1)
    grad_dist =doutput_ddist_expanded_summed
    
    return grad_dist

def der_act(x):
    sig = torch.nn.Sigmoid()
    sigx = sig(x)
    return sigx + x * sigx - x * sigx**2 

def layer_norm_backward(dY, X, gamma, eps):
    # Dimensions
    N = X.size(-1)
    dims = (-1,)

    # Compute mean and variance of X along the last dimension
    mean = X.mean(dims, keepdim=True)
    var = X.var(dims, keepdim=True, unbiased=False)
    rstd = 1 / torch.sqrt(var + eps)

    # Use default gamma value if not provided
    gamma_val = gamma if gamma is not None else torch.ones_like(X)

    # Calculate stats_x1 and stats_x2 using vectorized operations
    stats_x1 = (dY * gamma_val * rstd).sum(dims, keepdim=True)
    stats_x2 = ((dY * gamma_val) * (X - mean) * rstd).sum(dims, keepdim=True)

    # Compute gradients with respect to X using vectorized operations
    term1 = (1 / N) * rstd
    f_grad_input = N * gamma_val * dY
    f_grad_input -= (X - mean) * rstd * stats_x2
    f_grad_input -= (dY * gamma_val).sum(dims, keepdim=True)
    f_grad_input *= term1

    return f_grad_input

def grad_symtensor(grad_S, v):
    # Calculate the gradient of the outer product term
    grad_v_outer = torch.matmul(grad_S + (grad_S).transpose(-1,-2), v.unsqueeze(-1)).squeeze(-1)
    # Calculate the gradient of the diagonal reduction term
    grad_v_diag = 2/3 * v * grad_S.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
    # Combine the gradients
    grad_v = grad_v_outer - grad_v_diag

    return grad_v

def grad_skewtensor(grad_output, vector):
    grad_vector = torch.empty_like(vector)
    grad_vector[:, 0] = -grad_output[:, 1, 2] + grad_output[:, 2, 1]
    grad_vector[:, 1] = grad_output[:, 0, 2] - grad_output[:, 2, 0]
    grad_vector[:, 2] = -grad_output[:, 0, 1] + grad_output[:, 1, 0]
    return grad_vector



class TensorForceNet(nn.Module):
    
    def __init__(
        self,
        hidden_channels=128,
        num_layers=2,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        cutoff_lower=0,
        cutoff_upper=4.5,
        max_num_neighbors=64,
        max_z=128,
        equivariance_invariance_group="O(3)",
        static_shapes=True,
        dtype=torch.float32,
    ):
        super(TensorNet, self).__init__()

        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        assert equivariance_invariance_group in ["O(3)", "SO(3)"], (
            f'Unknown group "{equivariance_invariance_group}". '
            f"Choose O(3) or SO(3)."
        )
        self.hidden_channels = hidden_channels
        self.equivariance_invariance_group = equivariance_invariance_group
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        act_class = act_class_mapping[activation]
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )

        self.distance_proj1 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj2 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj3 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.max_z = max_z
        self.emb = nn.Embedding(max_z, hidden_channels, dtype=dtype)
        self.emb2 = nn.Linear(2 * hidden_channels, hidden_channels, dtype=dtype)
        self.act = activation()
        self.linears_tensor = nn.ModuleList()
        for _ in range(3):
            self.linears_tensor.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.init_norm = nn.LayerNorm(hidden_channels, dtype=dtype)

        
        self.linear = nn.Linear(3 * hidden_channels, hidden_channels, dtype=dtype)
        self.out_norm = nn.LayerNorm(3 * hidden_channels, dtype=dtype)
        self.output_linear1 = nn.Linear(hidden_channels, hidden_channels // 2, dtype=dtype)
        self.output_linear2 = nn.Linear(hidden_channels // 2, 1, dtype=dtype)
        self.act = act_class()
        
        self.static_shapes = static_shapes
        self.distance = OptimizedDistance(
            cutoff_lower,
            cutoff_upper,
            max_num_pairs=-max_num_neighbors,
            return_vecs=True,
            loop=True,
            check_errors=False,
            resize_to_fit=not self.static_shapes,
            long_edge_index=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()
        self.linear.reset_parameters()
        self.out_norm.reset_parameters()
        nn.init.xavier_uniform_(self.output_linear1.weight)
        self.output_linear1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_linear2.weight)
        self.output_linear2.bias.data.fill_(0)

    def _get_atomic_number_message(self, zp: Tensor, edge_index: Tensor) -> Tensor:
        Z = self.emb(zp)
        Zij = self.emb2(
            Z.index_select(0, edge_index.t().reshape(-1)).view(
                -1, self.hidden_channels * 2
            )
        )[..., None, None]
        return Zij

    def _get_tensor_messages(
        self, Zij: Tensor, edge_weight: Tensor, edge_vec_norm: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        C = self.cutoff(edge_weight).reshape(-1, 1, 1, 1) * Zij
        eye = torch.eye(3, 3, device=edge_vec_norm.device, dtype=edge_vec_norm.dtype)[
            None, None, ...
        ]
        distance_proj1 = self.distance_proj1(edge_attr)
        distance_proj2 = self.distance_proj2(edge_attr)
        distance_proj3 = self.distance_proj3(edge_attr)
        
        Iij = distance_proj1[..., None, None] * C * eye
        Aij = (
            distance_proj2[..., None, None]
            * C
            * vector_to_skewtensor(edge_vec_norm)[..., None, :, :]
        )
        Sij = (
            distance_proj3[..., None, None]
            * C
            * vector_to_symtensor(edge_vec_norm)[..., None, :, :]
        )
        return Iij, Aij, Sij, distance_proj1, distance_proj2, distance_proj3, C, eye

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Obtain graph, with distances and relative position vectors
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        # This assert convinces TorchScript that edge_vec is a Tensor and not an Optional[Tensor]
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"
        # Distance module returns -1 for non-existing edges, to avoid having to resize the tensors when we want to ensure static shapes (for CUDA graphs) we make all non-existing edges pertain to a ghost atom
        zp = z
        if self.static_shapes:
            mask = (edge_index[0] < 0).unsqueeze(0).expand_as(edge_index)
            zp = torch.cat((z, torch.zeros(1, device=z.device, dtype=z.dtype)), dim=0)
            # I trick the model into thinking that the masked edges pertain to the extra atom
            # WARNING: This can hurt performance if max_num_pairs >> actual_num_pairs
            edge_index = edge_index.masked_fill(mask, z.shape[0])
            edge_weight = edge_weight.masked_fill(mask[0], 0)
            edge_vec = edge_vec.masked_fill(mask[0].unsqueeze(-1).expand_as(edge_vec), 0)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] == edge_index[1]
        # Normalizing edge vectors by their length can result in NaNs, breaking Autograd.
        # I avoid dividing by zero by setting the weight of self edges and self loops to 1
        edge_vec_norm = edge_vec / edge_weight.masked_fill(mask, 1).unsqueeze(1)

        Zij = self._get_atomic_number_message(zp, edge_index)
        Iij, Aij, Sij, distance_proj1, distance_proj2, distance_proj3, C, eye = self._get_tensor_messages(
            Zij, edge_weight, edge_vec_norm, edge_attr
        )
        source = torch.zeros(
            zp.shape[0], self.hidden_channels, 3, 3, device=zp.device, dtype=Iij.dtype
        )
        I0 = source.index_add(dim=0, index=edge_index[0], source=Iij)
        A0 = source.index_add(dim=0, index=edge_index[0], source=Aij)
        S0 = source.index_add(dim=0, index=edge_index[0], source=Sij)

        tensornorm = tensor_norm(I0+A0+S0)
        tnorm = self.init_norm(tensornorm)
        norm1 = self.linears_scalar[0](tnorm)
        act_norm1 = self.act(norm1)
        norm2 = self.linears_scalar[1](act_norm1)
        norm = self.act(norm2)
        norm = norm.reshape(-1, self.hidden_channels, 3)
        
        I = (
            self.linears_tensor[0](I0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 0, None, None]
        )
        A = (
            self.linears_tensor[1](A0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 1, None, None]
        )
        S = (
            self.linears_tensor[2](S0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 2, None, None]
        )
        
        tnormI = tensor_norm(I)
        tnormA = tensor_norm(A)
        tnormS = tensor_norm(S)
        
        x = torch.cat((tnormI, tnormA, tnormS), dim=-1)
        s = self.out_norm(x)

        s1 = self.linear(s)
        act_s1 = self.act(s1)
        s2 = self.output_linear1(act_s1)
        act_s2 = self.act(s2)
        s3 = self.output_linear2(act_s2) 

        if self.static_shapes:
            s3 = s3[:-1]

        ## backward
        if self.static_shapes:
            grad_s = torch.cat((torch.ones_like(s3, device=s3.device, dtype=s3.dtype),
                                torch.zeros(1,1, device=s3.device, dtype=s3.dtype)),dim=0)
        else:
            grad_s = torch.ones_like(s3, device=s3.device, dtype=s3.dtype)
            
        grad_s = grad_s @ lin3.weight
        grad_s = (der_act(s2) * grad_s) @ lin2.weight
        grad_s = (der_act(s1) * grad_s) @ lin1.weight
        grad_x = layer_norm_backward(grad_s, x, self.out_norm.weight, self.out_norm.eps)
        grad_tnormI = grad_x[:,:self.hidden_channels]
        grad_tnormA = grad_x[:,self.hidden_channels:2*self.hidden_channels]
        grad_tnormS = grad_x[:,2*self.hidden_channels:]

        grad_I = 2 * I * grad_tnormI[...,None,None]
        grad_A = 2 * A * grad_tnormA[...,None,None]
        grad_S = 2 * S * grad_tnormS[...,None,None]

        der_act_norm2 = der_act(norm2).reshape(norm2.shape[0],self.hidden_channels,3)
        der_act_norm2[...,0] = der_act_norm2[...,0] * (I1 * grad_I).sum((-1,-2))
        der_act_norm2[...,1] = der_act_norm2[...,1] * (A1 * grad_A).sum((-1,-2))
        der_act_norm2[...,2] = der_act_norm2[...,2] * (S1 * grad_S).sum((-1,-2))
        der_act_norm2 = der_act_norm2.reshape(norm2.shape[0],self.hidden_channels * 3)

        grad_norm = ((der_act_norm2 @ self.linears_scalar[1].weight) * der_act(norm1)) @ self.linears_scalar[0].weight

        grad_tensornorm = layer_norm_backward(grad_norm, tensornorm, self.init_norm.weight, self.init_norm.eps)

        grad_I0 = ((norm[...,0,None,None]*grad_I).permute(0, 2, 3, 1) @ self.linears_tensor[0].weight).permute(0, 3, 1, 2)
        grad_I0 = grad_I0 + 2 * (I0+A0+S0) * grad_tensornorm[...,None,None]

        grad_A0 = ((norm[...,1,None,None]*grad_A).permute(0, 2, 3, 1) @ self.linears_tensor[1].weight).permute(0, 3, 1, 2)
        grad_A0 = grad_A0 + 2 * (I0+A0+S0) * grad_tensornorm[...,None,None]

        grad_S0 = ((norm[...,2,None,None]*grad_S).permute(0, 2, 3, 1) @ self.linears_tensor[2].weight).permute(0, 3, 1, 2)
        grad_S0 = grad_S0 + 2 * (I0+A0+S0) * grad_tensornorm[...,None,None]

        grad_Iij = grad_I0[edge_index[0]]
        grad_Aij = grad_A0[edge_index[0]]
        grad_Sij = grad_S0[edge_index[0]]

        grad_distance_proj1 = (grad_Iij * C * eye).sum((-1,-2))
        grad_distance_proj2 = (grad_Aij * C * vector_to_skewtensor(edge_vec_norm)[..., None, :, :]).sum((-1,-2))
        grad_distance_proj3 = (grad_Sij * C * vector_to_symtensor(edge_vec_norm)[..., None, :, :]).sum((-1,-2))

        grad_edge_attr = grad_distance_proj1 @ self.distance_proj1.weight + (
                    grad_distance_proj2 @ self.distance_proj2.weight + grad_distance_proj3 @ self.distance_proj3.weight)

        grad_C = distance_proj1[...,None,None] * eye * grad_Iij + (
            distance_proj2[...,None,None] * vector_to_skewtensor(edge_vec_norm)[..., None, :, :] * grad_Aij
            + distance_proj3[...,None,None] * vector_to_symtensor(edge_vec_norm)[..., None, :, :] * grad_Sij)

        grad_edge_weight = grad_rbfs(edge_weight, self.distance.betas, self.distance.means, self.distance.alpha,
                                     self.cutoff_upper, self.cutoff, grad_edge_attr)
        grad_edge_weight = grad_edge_weight + (grad_C * Zij).sum((-1,-2,-3)) * der_cutoff(edge_weight,1.1)

        grad_edge_vec_norm1 = grad_symtensor((distance_proj3[...,None,None] * C * grad_Sij).sum(-3), edge_vec_norm)
        grad_edge_vec_norm2 = grad_skewtensor((distance_proj2[...,None,None] * C * grad_Aij).sum(-3), edge_vec_norm)
        grad_edge_vec_norm = grad_edge_vec_norm1
        grad_edge_vec_norm = grad_edge_vec_norm + grad_edge_vec_norm2
        grad_edge_vec = grad_edge_vec_norm / edge_weight.masked_fill(mask, 1).unsqueeze(1)

        grad_edge_weight = grad_edge_weight + (1-mask.float()) * (
            -edge_vec * grad_edge_vec_norm/((edge_weight**2).masked_fill(mask, 1)).unsqueeze(-1)).sum(-1)

        num_atoms=z.shape[0]
        zero_mask = edge_weight == 0
        zero_mask3 = zero_mask.unsqueeze(-1).expand_as(grad_edge_vec)
        grad_distances_ = edge_vec / edge_weight.masked_fill(zero_mask, 1).unsqueeze(-1) * \
                          grad_edge_weight.masked_fill(zero_mask, 0).unsqueeze(-1)
        result = grad_edge_vec.masked_fill(zero_mask3, 0) + grad_distances_
        grad_positions_ = torch.zeros([num_atoms + 1, 3], dtype=edge_vec.dtype, device=edge_vec.device)
        edge_index_ = edge_index.masked_fill(zero_mask.unsqueeze(0).expand_as(edge_index), num_atoms)
        grad_positions_.index_add_(0, edge_index_[0], result)
        grad_positions_.index_add_(0, edge_index_[1], -result)
        grad_positions = grad_positions_[:num_atoms]
        
        return s3, grad_pos, z, pos, batch
