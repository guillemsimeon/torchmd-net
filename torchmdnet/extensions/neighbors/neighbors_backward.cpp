#include "neighbors_backwards.h"

using torch::indexing::Slice;

Tensor neighbors_backward(const Tensor& edge_index, const Tensor& edge_vec,
                          const Tensor& edge_weight, const Tensor& grad_edge_vec,
                          const Tensor& grad_edge_weight, int64_t num_atoms) {
    auto zero_mask = edge_weight == 0;
    auto zero_mask3 = zero_mask.unsqueeze(-1).expand_as(grad_edge_vec);
    // We need to avoid dividing by 0. Otherwise Autograd fills the gradient with NaNs in the
    // case of a double backwards. This is why we index_select like this.
    auto grad_distances_ = edge_vec / edge_weight.masked_fill(zero_mask, 1).unsqueeze(-1) *
                           grad_edge_weight.masked_fill(zero_mask, 0).unsqueeze(-1);
    auto result = grad_edge_vec.masked_fill(zero_mask3, 0) + grad_distances_;
    // Given that there is no masked_index_add function, in order to make the operation
    // CUDA-graph compatible I need to transform masked indices into a dummy value (num_atoms)
    // and then exclude that value from the output.
    // TODO: replace this once masked_index_add  or masked_scatter_add are available
    auto grad_positions_ = torch::zeros({num_atoms + 1, 3}, edge_vec.options());
    auto edge_index_ =
        edge_index.masked_fill(zero_mask.unsqueeze(0).expand_as(edge_index), num_atoms);
    grad_positions_.index_add_(0, edge_index_[0], result);
    grad_positions_.index_add_(0, edge_index_[1], -result);
    auto grad_positions = grad_positions_.index({Slice(0, num_atoms), Slice()});
    return grad_positions;
}
