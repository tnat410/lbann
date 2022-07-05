////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#define LBANN_ROTATION_LAYER_INSTANTIATE
#include "lbann/layers/image/cutout.hpp"

#include <math.h>
#include <algorithm>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rotation_layer<TensorDataType, Layout, Device>::fp_compute() {

  // Useful constants
  constexpr DataType zero = 0;
  constexpr DataType one = 1;
  // Input and output tensors
  const auto& local_input = this->get_local_prev_activations();
  auto& local_output = this->get_local_activations();

  // Tensor dimensions
  const auto& input_dims = this->get_input_dims(0);
  const auto& num_samples = local_input.Width();
  const El::Int num_channels = input_dims[0];
  const El::Int input_height = input_dims[1];
  const El::Int input_width = input_dims[2];

  // Get cutout length
  const auto& cutouts = this->get_local_prev_activations(1).Get(0);

  // RNG
  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_real_distribution<DataType> uni(zero,one);
  const El::Int col_start = uni(gen)*input_width;
  const El::Int row_start = uni(gen)*input_height;

  // Perform cutout for each input pixel 
  LBANN_OMP_PARALLEL_FOR_COLLAPSE4
  for (El::Int sample = 0; sample < num_samples; ++sample) {
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      for (El::Int output_row = 0; output_row < input_height; ++output_row) {
        for (El::Int output_col = 0; output_col < input_width; ++output_col) {


          const auto& cutout = cutouts.Get(0,sample);

          // Find input pixels near cutout point
          const auto input_col = static_cast<El::Int>(std::floor(output_col));
          const auto input_row = static_cast<El::Int>(std::floor(output_row));

          // Input and output pixels
          auto& pixel_output = local_output(channel * input_height * input_width
                                                + output_row * input_width
                                                + output_col,
                                                sample);

	  if((input_col >= col_start && input_col < col_start + cutout) && (input_row >= row_start && input_row < row_start + cutout)){
          	pixel_output = zero;
	  }
        }
      }
    }
  }
}

#define PROTO(T) \
  template class cutout_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
