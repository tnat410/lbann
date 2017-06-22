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
//
// lbann_dnn_mnist.cpp - DNN application for mnist
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_mnist.hpp"
#include "lbann/callbacks/lbann_callback_dump_weights.hpp"
#include "lbann/callbacks/lbann_callback_dump_activations.hpp"
#include "lbann/callbacks/lbann_callback_dump_gradients.hpp"
#include "lbann/lbann.hpp"

using namespace lbann;

/// Main function
int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, 42);

#ifdef EL_USE_GPU
  El::GemmUseGPU(32,32,32);
#endif

  try {
    // Get data files
    const string g_MNIST_TrainLabelFile = Input("--train-label-file",
                                          "MNIST training set label file",
                                          std::string("train-labels-idx1-ubyte"));
    const string g_MNIST_TrainImageFile = Input("--train-image-file",
                                          "MNIST training set image file",
                                          std::string("train-images-idx3-ubyte"));
    const string g_MNIST_TestLabelFile = Input("--test-label-file",
                                         "MNIST test set label file",
                                         std::string("t10k-labels-idx1-ubyte"));
    const string g_MNIST_TestImageFile = Input("--test-image-file",
                                         "MNIST test set image file",
                                         std::string("t10k-images-idx3-ubyte"));

    //determine if we're going to scale, subtract mean, etc;
    //scaling/standardization is on a per-example basis (computed independantly
    //for each image)
    bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
    bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", false);
    bool unit_variance = Input("--unit-variance", "standardize to unit-variance", false);

    //if set to true, above three settings have no effect
    bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

    ///////////////////////////////////////////////////////////////////
    // initalize grid, block
    ///////////////////////////////////////////////////////////////////

    // Initialize parameter defaults
    TrainingParams trainParams;
    trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/MNIST/";
    trainParams.EpochCount = 20;
    trainParams.MBSize = 128;
    trainParams.LearnRate = 0.01;
    trainParams.DropOut = -1.0f;
    trainParams.ProcsPerModel = 0;
    trainParams.PercentageTrainingSamples = 1.0;
    trainParams.PercentageValidationSamples = 0.1;
    PerformanceParams perfParams;
    perfParams.BlockSize = 256;

    // Parse command-line inputs
    trainParams.parse_params();
    perfParams.parse_params();

    ProcessInput();
    PrintInputReport();

    // Set algorithmic blocksize
    SetBlocksize(perfParams.BlockSize);

    // Set up the communicator and get the grid.
    comm->split_models(trainParams.ProcsPerModel);
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      std::cout << "Number of models: " << comm->get_num_models() << std::endl;
      std::cout << "Grid is " << grid.Height() << " x " << grid.Width() << std::endl;
      std::cout << std::endl;
    }

    int parallel_io = perfParams.MaxParIOSize;
    if (parallel_io == 0) {
      if (comm->am_world_master()) {
        std::cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() <<
             " (Limited to # Processes)" << std::endl;
      }
      parallel_io = comm->get_procs_per_model();
    } else {
      if (comm->am_world_master()) {
        std::cout << "\tMax Parallel I/O Fetch: " << parallel_io << std::endl;
      }
    }

    ///////////////////////////////////////////////////////////////////
    // load training data (MNIST)
    mnist_reader mnist_trainset(trainParams.MBSize, true);
    mnist_trainset.set_file_dir(trainParams.DatasetRootDir);
    mnist_trainset.set_data_filename(g_MNIST_TrainImageFile);
    mnist_trainset.set_label_filename(g_MNIST_TrainLabelFile);
    mnist_trainset.set_validation_percent(trainParams.PercentageValidationSamples);
    mnist_trainset.load();

    mnist_trainset.scale(scale);
    mnist_trainset.subtract_mean(subtract_mean);
    mnist_trainset.unit_variance(unit_variance);
    mnist_trainset.z_score(z_score);

    ///////////////////////////////////////////////////////////////////
    // create a validation set from the unused training data (MNIST)
    ///////////////////////////////////////////////////////////////////
    mnist_reader mnist_validation_set(mnist_trainset); // Clone the training set object
    mnist_validation_set.use_unused_index_set();

    if (comm->am_world_master()) {
      size_t num_train = mnist_trainset.getNumData();
      size_t num_validate = mnist_trainset.getNumData();
      double validate_percent = num_validate / (num_train+num_validate)*100.0;
      double train_percent = num_train / (num_train+num_validate)*100.0;
      std::cout << "Training using " << train_percent << "% of the training data set, which is " << mnist_trainset.getNumData() << " samples." << std::endl
           << "Validating training using " << validate_percent << "% of the training data set, which is " << mnist_validation_set.getNumData() << " samples." << std::endl;
    }

    ///////////////////////////////////////////////////////////////////
    // load testing data (MNIST)
    ///////////////////////////////////////////////////////////////////
    mnist_reader mnist_testset(trainParams.MBSize, true);
    mnist_testset.set_file_dir(trainParams.DatasetRootDir);
    mnist_testset.set_data_filename(g_MNIST_TestImageFile);
    mnist_testset.set_label_filename(g_MNIST_TestLabelFile);
    mnist_testset.set_use_percent(trainParams.PercentageTestingSamples);
    mnist_testset.load();

    mnist_testset.scale(scale);
    mnist_testset.subtract_mean(subtract_mean);
    mnist_testset.unit_variance(unit_variance);
    mnist_testset.z_score(z_score);

    if (comm->am_world_master()) {
      std::cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << mnist_testset.getNumData() << " samples." << std::endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize neural network (layers)
    ///////////////////////////////////////////////////////////////////

    // Initialize optimizer
    optimizer_factory *optimizer_fac;
    if (trainParams.LearnRateMethod == 1) { // Adagrad
      optimizer_fac = new adagrad_factory(comm, trainParams.LearnRate);
      cout << "XX adagrad\n";
    } else if (trainParams.LearnRateMethod == 2) { // RMSprop
      optimizer_fac = new rmsprop_factory(comm, trainParams.LearnRate);
      cout << "XX rmsprop\n";
    } else if (trainParams.LearnRateMethod == 3) { // Adam
      optimizer_fac = new adam_factory(comm, trainParams.LearnRate);
      cout << "XX adam\n";
    } else {
      optimizer_fac = new sgd_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
      cout << "XX sgd\n";
    }

    // Initialize network
    deep_neural_network dnn(trainParams.MBSize, comm, new objective_functions::mean_squared_error(comm),optimizer_fac);
    dnn.add_metric(new metrics::categorical_accuracy(data_layout::MODEL_PARALLEL, comm));
    std::map<execution_mode, generic_data_reader *> data_readers = {std::make_pair(execution_mode::training,&mnist_trainset),
                                                           std::make_pair(execution_mode::validation, &mnist_validation_set),
                                                           std::make_pair(execution_mode::testing, &mnist_testset)
                                                          };


    Layer *input_layer = new input_layer_distributed_minibatch_parallel_io<data_layout::MODEL_PARALLEL>(comm, parallel_io, (int) trainParams.MBSize, data_readers);
    dnn.add(input_layer);

    Layer *encode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(1,
                                                        mnist_trainset.get_linearized_data_size(), 1000,trainParams.MBSize,
                                                        weight_initialization::uniform, comm,
                                                        optimizer_fac->create_optimizer());
    dnn.add(encode1);

    Layer *tanh1 = new tanh_layer<data_layout::MODEL_PARALLEL>(2, comm,
                                               trainParams.MBSize, 1000);
    dnn.add(tanh1);

    /*Layer *dropout1 = new dropout<data_layout::MODEL_PARALLEL>(3, 1000,
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);
    dnn.add(dropout1);*/

    //third layer 
    Layer *encode2 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(3,
                                                        1000, 500, trainParams.MBSize,
                                                        weight_initialization::uniform, 
                                                        comm, optimizer_fac->create_optimizer());
    dnn.add(encode2);

    Layer *tanh2 = new tanh_layer<data_layout::MODEL_PARALLEL>(4, comm,
                                               trainParams.MBSize, 500);
    dnn.add(tanh2);

    /*Layer *dropout2 = new dropout<data_layout::MODEL_PARALLEL>(6, 500,
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);
    dnn.add(dropout2);*/

    Layer *encode3 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(5,
                                                        500, 250, trainParams.MBSize,
                                                        weight_initialization::uniform,
                                                        comm, optimizer_fac->create_optimizer());
    dnn.add(encode3);
    
    Layer *tanh3 = new tanh_layer<data_layout::MODEL_PARALLEL>(6, comm,
                                               trainParams.MBSize, 250);
    dnn.add(tanh3);

    /*Layer *dropout3 = new dropout<data_layout::MODEL_PARALLEL>(9, 250,
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);
    dnn.add(dropout3);*/

    Layer *encode4 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(7,
                                                        250, 30, trainParams.MBSize,
                                                        weight_initialization::uniform,
                                                        comm, optimizer_fac->create_optimizer());
    dnn.add(encode4);
    //decoder
    Layer *decode4 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(8,
                                                        30, 250, trainParams.MBSize,
                                                        weight_initialization::uniform,
                                                        comm, optimizer_fac->create_optimizer());
    dnn.add(decode4);
    
    Layer *tanh4 = new tanh_layer<data_layout::MODEL_PARALLEL>(9, comm,
                                               trainParams.MBSize, 250);
    dnn.add(tanh4);

    /*Layer *dropout4 = new dropout<data_layout::MODEL_PARALLEL>(13, 250,
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);
    dnn.add(dropout4);*/

   
    Layer *decode3 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(10,
                                                      250, 500, trainParams.MBSize,
                                                        weight_initialization::uniform,
                                                        comm, optimizer_fac->create_optimizer());
    dnn.add(decode3);
    
    Layer *tanh5 = new tanh_layer<data_layout::MODEL_PARALLEL>(11, comm,
                                               trainParams.MBSize, 500);
    dnn.add(tanh5);

    /*Layer *dropout5 = new dropout<data_layout::MODEL_PARALLEL>(16, 500,
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);
    dnn.add(dropout5);*/

    Layer *decode2 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(12,
                                                        500,1000, trainParams.MBSize,
                                                        weight_initialization::uniform,
                                                        comm, optimizer_fac->create_optimizer());
    dnn.add(decode2);
    
    Layer *tanh6 = new tanh_layer<data_layout::MODEL_PARALLEL>(13, comm,
                                               trainParams.MBSize, 1000);
    dnn.add(tanh6);

    /*Layer *dropout6 = new dropout<data_layout::MODEL_PARALLEL>(14, 1000,
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);
    dnn.add(dropout6);*/

    Layer *decode1 = new fully_connected_layer<data_layout::MODEL_PARALLEL>(14,
                                                        1000, 784, trainParams.MBSize,
                                                        weight_initialization::uniform,
                                                        comm, optimizer_fac->create_optimizer());
    dnn.add(decode1);
    
    Layer *tanh7 = new tanh_layer<data_layout::MODEL_PARALLEL>(15, comm,
                                               trainParams.MBSize, 784);
    dnn.add(tanh7);

    /*Layer *dropout7 = new dropout<data_layout::MODEL_PARALLEL>(22, 784,
                                               comm, trainParams.MBSize,
                                               trainParams.DropOut);
    dnn.add(dropout7);*/

    Layer* rcl  = new reconstruction_layer<data_layout::MODEL_PARALLEL>(16, comm, 
                                                          optimizer_fac->create_optimizer(), 
                                                          trainParams.MBSize, input_layer);
    dnn.add(rcl);

    lbann_callback_dump_weights *dump_weights_cb = nullptr;
    lbann_callback_dump_activations *dump_activations_cb = nullptr;
    lbann_callback_dump_gradients *dump_gradients_cb = nullptr;
    if (trainParams.DumpWeights) {
      dump_weights_cb = new lbann_callback_dump_weights(
        trainParams.DumpDir);
      dnn.add_callback(dump_weights_cb);
    }
    if (trainParams.DumpActivations) {
      dump_activations_cb = new lbann_callback_dump_activations(
        trainParams.DumpDir);
      dnn.add_callback(dump_activations_cb);
    }
    if (trainParams.DumpGradients) {
      dump_gradients_cb = new lbann_callback_dump_gradients(
        trainParams.DumpDir);
      dnn.add_callback(dump_gradients_cb);
    }

    if (comm->am_world_master()) {
      std::cout << "Parameter settings:" << std::endl;
      std::cout << "\tMini-batch size: " << trainParams.MBSize << std::endl;
      std::cout << "\tLearning rate: " << trainParams.LearnRate << std::endl << std::endl;
      std::cout << "\tEpoch count: " << trainParams.EpochCount << std::endl;
    }

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

    // Initialize the model's data structures
    dnn.setup();

    // set checkpoint directory and checkpoint interval
    dnn.set_checkpoint_dir(trainParams.ParameterDir);
    dnn.set_checkpoint_epochs(trainParams.CkptEpochs);
    dnn.set_checkpoint_steps(trainParams.CkptSteps);
    dnn.set_checkpoint_secs(trainParams.CkptSecs);

    // restart model from checkpoint if we have one
    dnn.restartShared();

    // train/test
    while (dnn.get_cur_epoch() < trainParams.EpochCount) {
      dnn.train(1, true);
      // testing
      dnn.evaluate(execution_mode::testing);
    }

    // Free dynamically allocated memory
    // delete target_layer;  // Causes segfault
    // delete input_layer;  // Causes segfault
    // delete lfac;  // Causes segfault
    if (trainParams.DumpWeights) {
      delete dump_weights_cb;
    }
    if (trainParams.DumpActivations) {
      delete dump_activations_cb;
    }
    if (trainParams.DumpGradients) {
      delete dump_gradients_cb;
    }
    delete optimizer_fac;
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  finalize(comm);

  return 0;
}