//@HEADER
// ************************************************************************
// Copyright 2018 National Technology & Engineering Solutions of Sandia,
// LLC (NTESS).  Under the terms of Contract DE-NA0003525, the U.S. 
// Government retains certain rights // in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions 
// are met:
//
// 1. Redistributions of source code must retain the above copyright 
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright 
//    notice, this list of conditions and the following disclaimer in 
//    the documentation and/or other materials provided with the 
//    distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER 
// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are 
// those of the authors and should not be interpreted as representing 
// official policies, either expressed or implied, of NTESS or the U.S. 
// Government.
// ************************************************************************
//@HEADER

#include <cstdlib>

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

#include "AnalyzeConfig.hpp"
#include "alg/Run.hpp"

#include "alg/ErrorHandling.hpp"
#include "PlatoDriver.hpp"

namespace Plato
{

#if defined(KOKKOS_HAVE_CUDA)
void run_cuda_query(comm::Machine machine)
{
    const size_t comm_rank = comm::rank(machine);
    std::cout << "P" << comm_rank
    << ": Cuda device_count = " << Kokkos::Cuda::detect_device_count()
    << std::endl;
}
#endif

void run(Omega_h::Library* aOmegaHLib, Teuchos::ParameterList& aProblem, comm::Machine aMachine)
{
    if(comm::rank(aMachine) == 0)
    {

        std::cout << "\nRunning Plato Analyze version " << version_major << "."
                << version_minor << "." << version_patch << std::endl;

        std::cout << "\n\nparameter list:" << std::endl;
        aProblem.print(std::cout);
    }

    auto tOutput_viz = aProblem.get<std::string>("Output Viz");
    auto tInput_mesh = aProblem.get<std::string>("Input Mesh");

    auto physicsString = aProblem.get<std::string>("Physics", "Default");
    if(aProblem.get<bool>("Query"))
    {
        if(comm::rank(aMachine) == 0)
        {
            const unsigned numa_count = Kokkos::hwloc::get_available_numa_count();
            const unsigned cores_per_numa = Kokkos::hwloc::get_available_cores_per_numa();
            const unsigned threads_per_core = Kokkos::hwloc::get_available_threads_per_core();
            std::cout << "P" << comm::rank(aMachine) << ": hwloc { NUMA[" << numa_count << "]" << " CORE[" << cores_per_numa
                      << "]" << " PU[" << threads_per_core << "] }" << std::endl;
        }
#if defined(KOKKOS_HAVE_CUDA)
        Plato::run_cuda_query(machine);
#endif
    }
    else
    {
        if(physicsString == "Plato Driver")
        {
            ::Plato::driver(aOmegaHLib, aProblem, tInput_mesh, tOutput_viz);
        }
        else
        {
            THROWERR("Unrecognized Physics " << physicsString);
        }
    }
}
}
//----------------------------------------------------------------------------
