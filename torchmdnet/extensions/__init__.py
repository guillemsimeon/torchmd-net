# Place here any short extensions to torch that you want to use in your code.
# The extensions present in extensions.cpp will be automatically compiled in setup.py and loaded here.
# The extensions will be available under torch.ops.torchmdnet_extensions, but you can add wrappers here to make them more convenient to use.
import os.path as osp
import torch
from torch import Tensor
import importlib.machinery
from typing import Tuple, Optional, List


def _load_library(library):
    """Load a dynamic library containing torch extensions from the given path.
    Args:
        library (str): The name of the library to load.
    """
    # Find the specification for the library
    spec = importlib.machinery.PathFinder().find_spec(library, [osp.dirname(__file__)])
    # Check if the specification is found and load the library
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:
        raise ImportError(
            f"Could not find module '{library}' in {osp.dirname(__file__)}"
        )


_load_library("torchmdnet_extensions")


def is_current_stream_capturing():
    """Returns True if the current CUDA stream is capturing.

    Returns False if CUDA is not available or the current stream is not capturing.

    This utility is required because the builtin torch function that does this is not scriptable.
    """
    _is_current_stream_capturing = (
        torch.ops.torchmdnet_extensions.is_current_stream_capturing
    )
    return _is_current_stream_capturing()


def neighbor_forward(
    strategy: str,
    positions: Tensor,
    batch: Tensor,
    box_vectors: Tensor,
    use_periodic: bool,
    cutoff_lower: float,
    cutoff_upper: float,
    max_num_pairs: int,
    loop: bool,
    include_transpose: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the neighbor pairs for a given set of atomic positions.

    The list is generated as a list of pairs (i,j) without any enforced ordering.
    The list is padded with -1 to the maximum number of pairs.

    Parameters
    ----------
    strategy : str
        Strategy to use for computing the neighbor list. Can be one of :code:`["shared", "brute", "cell"]`.
    positions : Tensor
        A tensor with shape (N, 3) representing the atomic positions.
    batch : Tensor
        A tensor with shape (N,). Specifies the batch for each atom.
    box_vectors : Tensor
        The vectors defining the periodic box with shape `(3, 3)`.
    use_periodic : bool
        Whether to apply periodic boundary conditions.
    cutoff_lower : float
        Lower cutoff for the neighbor list.
    cutoff_upper : float
        Upper cutoff for the neighbor list.
    max_num_pairs : int
        Maximum number of pairs to store.
    loop : bool
        Whether to include self-interactions.
    include_transpose : bool
        Whether to include the transpose of the neighbor list (pair i,j and pair j,i).

    Returns
    -------
    neighbors : Tensor
        List of neighbors for each atom. Shape (2, max_num_pairs).
    distances : Tensor
        List of distances for each atom. Shape (max_num_pairs,).
    distance_vecs : Tensor
        List of distance vectors for each atom. Shape (max_num_pairs, 3).
    num_pairs : Tensor
        The number of pairs found.

    Notes
    -----
    - This function is a torch extension loaded from `torch.ops.torchmdnet_extensions.get_neighbor_pairs`.
    """
    return torch.ops.torchmdnet_extensions.get_neighbor_pairs(
        strategy,
        positions,
        batch,
        box_vectors,
        use_periodic,
        cutoff_lower,
        cutoff_upper,
        max_num_pairs,
        loop,
        include_transpose,
    )


def neighbor_backward(
    edge_index: Tensor,
    edge_vec: Tensor,
    edge_weight: Tensor,
    num_atoms: int,
    grad_edge_vec: Optional[Tensor] = None,
    grad_edge_weight: Optional[Tensor] = None,
) -> Tensor:
    """Computes the neighbor pairs for a given set of atomic positions. This is the backwards pass of the :any:`get_neighbor_pairs_kernel` function.

    Parameters
    ----------
    edge_index : Tensor
        A tensor with shape (2, max_num_pairs) representing the neighbor pairs.
    edge_vec : Tensor
        A tensor with shape (max_num_pairs, 3) representing the distance vectors.
    edge_weight : Tensor
        A tensor with shape (max_num_pairs,) representing the distances.
    num_atoms : int
        The number of atoms.
    grad_edge_vec : Tensor, optional
        The gradient of the distance vectors. If None, the gradient is assumed to be 1.
    grad_edge_weight : Tensor, optional
        The gradient of the distances. If None, the gradient is assumed to be 1.

    Returns
    -------
    grad_positions : Tensor
        The gradient of the positions. Shape (N, 3).
    """
    if grad_edge_vec is None:
        grad_edge_vec = torch.ones_like(edge_vec)
    if grad_edge_weight is None:
        grad_edge_weight = torch.ones_like(edge_weight)
    return torch.ops.torchmdnet_extensions.get_neighbor_pairs_backward(
        edge_index, edge_vec, edge_weight, grad_edge_vec, grad_edge_weight, num_atoms
    )


# # For some unknown reason torch.compile is not able to compile this function
# if int(torch.__version__.split(".")[0]) >= 2:
#     import torch._dynamo as dynamo

#     dynamo.disallow_in_graph(get_neighbor_pairs_kernel)


# This class is a PyTorch autograd extension for computing neighbor pairs and their gradients.
class get_neighbor_pairs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        strategy: str,
        positions: Tensor,
        batch: Tensor,
        box_vectors: Tensor,
        use_periodic: bool,
        cutoff_lower: float,
        cutoff_upper: float,
        max_num_pairs: int,
        loop: bool,
        include_transpose: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Call the forward kernel and store the results
        neighbors, deltas, distances, i_curr_pair = neighbor_forward(
            strategy,
            positions,
            batch,
            box_vectors,
            use_periodic,
            cutoff_lower,
            cutoff_upper,
            max_num_pairs,
            loop,
            include_transpose,
        )
        # Save tensors for backward computation
        ctx.save_for_backward(neighbors, deltas, distances)
        ctx.num_atoms = positions.size(0)
        return (neighbors, deltas, distances, i_curr_pair)

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tensor
    ) -> Tuple[
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        neighbors, deltas, distances = ctx.saved_tensors
        num_atoms = ctx.num_atoms
        grad_edge_vec = grad_outputs[1]
        grad_edge_weight = grad_outputs[2]
        grad_positions = neighbor_backward(
            neighbors, deltas, distances, num_atoms, grad_edge_vec, grad_edge_weight
        )
        return (
            None,
            grad_positions,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def get_neighbor_pairs_kernel(
    strategy: str,
    positions: Tensor,
    batch: Tensor,
    box_vectors: Tensor,
    use_periodic: bool,
    cutoff_lower: float,
    cutoff_upper: float,
    max_num_pairs: int,
    loop: bool,
    include_transpose: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return get_neighbor_pairs.apply(
        strategy,
        positions,
        batch,
        box_vectors,
        use_periodic,
        cutoff_lower,
        cutoff_upper,
        max_num_pairs,
        loop,
        include_transpose,
    )
