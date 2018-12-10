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
////////////////////////////////////////////////////////////////////////////////

#include "lbann_config.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_hdf5.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include "lbann/utils/jag_utils.hpp"
#include <time.h>

using namespace lbann;


void get_scalar_names(std::vector<std::string> &s); 

void get_input_names(std::vector<std::string> &s); 

//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    if (!(opts->has_string("filelist") && opts->has_string("output_dir"))) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string> --output_dir=<string>");
      }
    }

    const std::string dir = opts->get_string("output_dir");

    if (master) {
      std::stringstream s;
      s << "mkdir -p " << opts->get_string("output_dir");
      int r = system(s.str().c_str());
      if (r != 0) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s.str());
      }
    }

    std::vector<std::string> files;
    const std::string fn = opts->get_string("filelist"); 
    read_filelist(comm, fn, files);
#if 0
    const int rank = comm->get_rank_in_world();
    int size;
    if (!rank) {
      std::stringstream s;
      std::ifstream in(opts->get_string("filelist").c_str());
      if (!in) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + opts->get_string("filelist") + " for reading");
      }
      std::string line;
      while (getline(in, line)) {
        if (line.size()) {
          s << line << " ";
        }
      }
      in.close();
      f = s.str();
      size = s.str().size();
      std::cout << "size: " << size << "\n";
    }
    comm->world_broadcast<int>(0, &size, 1);
    f.resize(size);
    comm->world_broadcast<char>(0, &f[0], size);

    std::stringstream s2(f);
    std::string filename;
    while (s2 >> filename) {
      if (filename.size()) {
        files.push_back(filename);
      }
    }
    if (rank==1) std::cerr << "num files: " << files.size() << "\n";
#endif
    //=======================================================================

    std::vector<std::string> scalar_names;
    std::vector<std::string> input_names;
    get_scalar_names(scalar_names);
    get_input_names(input_names);

    std::stringstream index;
    if (master) {
      index << "num_scalars: " << scalar_names.size() << "\n"
            << "num_inputs: " << input_names.size() << "\n"
            << "scalars: \n";
      for (auto t : scalar_names) {
        index << "        " << t << "\n";
      }
      index << "inputs:\n";
      for (auto t : input_names) {
        index << "        " << t << "\n";
      }
    }

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;

    if (master) std::cout << np << hdf5_file_hnd << "\n";

    int num_samples = 0;

    char b[1024];
    sprintf(b, "%s/tmp.%d", dir.c_str(), rank);
    std::ofstream out(b, std::ios::out | std::ios::binary);
    if (!out) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + b + " for writing");
    }

    size_t h = 0;
    for (size_t j=rank; j<files.size(); j+= np) {
      h += 1;
      if (h % 10 == 0) std::cout << rank << " :: processed " << h << " files\n";

      try {
std::cerr << rank << " :: opening for reading: " << files[j] << "\n";
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( files[j].c_str() );
      } catch (std::exception e) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << files[j] << "\n"; 
        continue;
      } catch (...) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << files[j] << "\n"; 
        continue;
      }  

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (std::exception e) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names; " << files[j] << "\n";
        continue;
      } catch (...) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names; " << files[j] << "\n";
        continue;
      }
std::cerr << rank << " :: num samples: " << cnames.size() << "\n";

      for (size_t i=0; i<cnames.size(); i++) {

        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (std::exception e) {
          std::cerr << rank << " :: exception reading success flag: " << files[j] << "\n";
          continue;
        } catch (...) {  
          std::cerr << rank << " :: exception reading success flag: " << files[j] << "\n";
          continue;
        }  

        int success = n_ok.to_int64();
        if (success == 1) {
          for (auto t : scalar_names) {
            key = cnames[i] + "/outputs/scalars/" + t;
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            double d = tmp.value();
            out.write((char*)&d, sizeof(double));
          }  

          for (auto t : input_names) {
            key = cnames[i] + "/inputs/" + t;
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            double d = tmp.value();
            out.write((char*)&d, sizeof(double));
          }
          ++num_samples;
        }
      }
    }
    out.close();

    comm->global_barrier();

    if (master) {
      std::stringstream s3;
      s3 << "cat " << dir << "/tmp* > " << dir << "/data.bin; rm -f " << dir << "/tmp*";
      int r = system(s3.str().c_str());
      if (r != 0) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s3.str());
      }
    }

    int global_num_samples;
    //int global_num_samples = comm->reduce<int>(num_samples, comm->get_world_comm());
    MPI_Reduce(&num_samples, &global_num_samples, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 

    if (master) {
      std::ofstream out2(dir + "/index.txt");
      if (!out2) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + dir  + "/index.txt for writing");
      }
      out2 << "num_samples: " << global_num_samples << "\n"
          << index.str();
      out2.close();

    }


  } catch (exception const &e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  }

  // Clean up
  finalize(comm);
  return EXIT_SUCCESS;
}


void get_input_names(std::vector<std::string> &s) {
  s.push_back("shape_model_initial_modes:(4,3)"); 
  s.push_back("betti_prl15_trans_u"); 
  s.push_back("betti_prl15_trans_v"); 
  s.push_back("shape_model_initial_modes:(2,1)"); 
  s.push_back("shape_model_initial_modes:(1,0)"); 
}

void get_scalar_names(std::vector<std::string> &s) {
  s.push_back("BWx");
  s.push_back("BT");
  s.push_back("tMAXt");
  s.push_back("BWn");
  s.push_back("MAXpressure");
  s.push_back("BAte");
  s.push_back("MAXtion");
  s.push_back("tMAXpressure");
  s.push_back("BAt");
  s.push_back("Yn");
  s.push_back("Ye");
  s.push_back("Yx");
  s.push_back("tMAXte");
  s.push_back("BAtion");
  s.push_back("MAXte");
  s.push_back("tMAXtion");
  s.push_back("BTx");
  s.push_back("MAXt");
  s.push_back("BTn");
  s.push_back("BApressure");
  s.push_back("tMINradius");
  s.push_back("MINradius");
}


#endif //#ifdef LBANN_HAS_CONDUIT
