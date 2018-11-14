////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYER_ACTIVATION_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/utils/cudnn.hpp"

// Threshold outputs to a minimum value.
// If enabled, the minimum output value is sqrt(min), where min is the
// minimum, normalized, positive value (~1e-19 for float and ~1e-154
// for double). The gradients w.r.t. input will be inaccurate, on the
// order of the minimum output value.
#define LBANN_ENABLE_SOFTMAX_CUTOFF

namespace lbann {

/** Softmax layer.
 *  \f[ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} \f]
 */
template <data_layout Layout, El::Device Device>
class softmax_layer : public Layer {
public:

  softmax_layer(lbann_comm *comm)
    : Layer(comm)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {}

  softmax_layer(const softmax_layer& other)
    : Layer(other),
      m_workspace(other.m_workspace ?
                  other.m_workspace->Copy() : nullptr)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
  }

  softmax_layer& operator=(const softmax_layer& other) {
    Layer::operator=(other);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() : nullptr);
#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~softmax_layer() = default;

  softmax_layer* copy() const override { return new softmax_layer(*this); }
  std::string get_type() const override { return "softmax"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_matrices(const El::Grid& grid) override {
    Layer::setup_matrices(grid);
    auto dist = get_prev_activations().DistData();
    dist.colDist = El::STAR;
    m_workspace.reset(AbsDistMat::Instantiate(dist));
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    Layer::fp_setup_outputs(mini_batch_size);
    const auto& dist_data = get_prev_activations().DistData();
    m_workspace->Empty(false);
    m_workspace->AlignWith(dist_data);
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Workspace for column-wise reductions. */
  std::unique_ptr<AbsDistMat> m_workspace;

#ifdef LBANN_HAS_CUDNN
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

};

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_SOFTMAX_HPP_INCLUDED
