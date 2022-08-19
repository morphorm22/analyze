#include "Teuchos_UnitTestRepository.hpp"
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "PlatoTestHelpers.hpp"

#ifdef WATCH_ARITHMETIC
#include <fenv.h>
#endif

int main( int argc, char* argv[] )
{

  MPI_Init(&argc, &argv);

#ifdef WATCH_ARITHMETIC
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_ALL_EXCEPT - FE_INEXACT - FE_UNDERFLOW);
#endif

  Kokkos::initialize(argc, argv);
  Plato::MeshFactory::initialize(argc, argv);

  auto result = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

  Plato::MeshFactory::finalize();
  Kokkos::finalize();
  MPI_Finalize();

  return result;
}
