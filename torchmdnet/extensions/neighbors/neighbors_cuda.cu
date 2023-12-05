/* Raul P. Pelaez 2023
   Connection between the neighbor CUDA implementations and the torch extension.
 */
#include "neighbors_backwards.h"
#include "neighbors_cuda_brute.cuh"
#include "neighbors_cuda_cell.cuh"
#include "neighbors_cuda_shared.cuh"
#include <torch/extension.h>

TORCH_LIBRARY_IMPL(torchmdnet_extensions, AutogradCUDA, m) {
    m.impl("get_neighbor_pairs",
           [](const std::string& strategy, const Tensor& positions, const Tensor& batch,
              const Tensor& box_vectors, bool use_periodic, const Scalar& cutoff_lower,
              const Scalar& cutoff_upper, const Scalar& max_num_pairs, bool loop,
              bool include_transpose) {
               auto final_strategy = strategy;
               if (positions.size(0) >= 32768 && strategy == "brute") {
                   final_strategy = "shared";
               }
               auto kernel = forward_brute;
               if (final_strategy == "cell") {
                   kernel = forward_cell;
               } else if (final_strategy == "shared") {
                   kernel = forward_shared;
               } else if (final_strategy != "brute") {
                   throw std::runtime_error("Unknown kernel name");
               }
               auto result = kernel(positions, batch, box_vectors, use_periodic, cutoff_lower,
                                    cutoff_upper, max_num_pairs, loop, include_transpose);
               return result;
           });
}
