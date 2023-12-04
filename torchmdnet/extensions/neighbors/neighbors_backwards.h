#include <torch/extension.h>
using torch::Tensor;

Tensor neighbors_backward(const Tensor& edge_index, const Tensor& edge_vec,
			  const Tensor& edge_weight, const Tensor& grad_edge_vec,
			  const Tensor& grad_edge_weight, int64_t num_atoms);
