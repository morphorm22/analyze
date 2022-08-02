#pragma once

#include <vector>

#include "Tacho.hpp"
#include "Tacho_Driver.hpp"
#include "Teuchos_ScalarTraits.hpp"

namespace tacho {

enum TACHO_PARAM_INDICES {
  USEDEFAULTSOLVERPARAMETERS,
  VERBOSITY,
  SMALLPROBLEMTHRESHOLDSIZE,
  SOLUTION_METHOD,
  TASKING_OPTION_BLOCKSIZE,
  TASKING_OPTION_PANELSIZE,
  TASKING_OPTION_MAXNUMSUPERBLOCKS,
  LEVELSET_OPTION_SCHEDULING,
  LEVELSET_OPTION_DEVICE_LEVEL_CUT,
  LEVELSET_OPTION_DEVICE_FACTOR_THRES,
  LEVELSET_OPTION_DEVICE_SOLVE_THRES,
  LEVELSET_OPTION_NSTREAMS,
  LEVELSET_OPTION_VARIANT,
  INDEX_LENGTH
};

void getTachoParams(std::vector<int> & tachoParams, const int solutionMethod=1);

using device_type =
    typename Tacho::UseThisDevice<Kokkos::DefaultExecutionSpace>::device_type;
using exec_space = typename device_type::execution_space;

using host_device_type = typename Tacho::UseThisDevice<
    Kokkos::DefaultHostExecutionSpace>::device_type;
using host_space = typename host_device_type::execution_space;

using ViewVectorType = Kokkos::View<double *, device_type>;
using ViewVectorTypeInt = Kokkos::View<int *, device_type>;

template <class SX> class tachoSolver {
public:
  using SM = typename Teuchos::ScalarTraits<SX>::magnitudeType;

  using device_type =
      typename Tacho::UseThisDevice<Kokkos::DefaultExecutionSpace>::type;
  typedef Tacho::Driver<SX, device_type> solver_type;

  typedef Tacho::ordinal_type ordinal_type;
  typedef Tacho::size_type size_type;

  typedef typename solver_type::value_type value_type;
  typedef typename solver_type::ordinal_type_array ordinal_type_array;
  typedef typename solver_type::size_type_array size_type_array;
  typedef typename solver_type::value_type_array value_type_array;
  typedef typename solver_type::value_type_matrix value_type_matrix;

  // host typedefs
  typedef typename solver_type::ordinal_type_array_host ordinal_type_array_host;
  typedef typename solver_type::size_type_array_host size_type_array_host;
  typedef typename solver_type::value_type_array_host value_type_array_host;
  typedef typename solver_type::value_type_matrix_host value_type_matrix_host;

  tachoSolver(const int *solverParams);
  ~tachoSolver();

  void refactorMatrix(const int numTerms, SX *values);

  int Initialize(int numRows,
                 /// with TACHO_ENABLE_INT_INT, size_type is "int"
                 int *rowBegin, int *columns, SX *values,
                 const bool printTimings = false);

  void MySolve(int NRHS, value_type_matrix &b, value_type_matrix &x);

private:
  int m_numRows;
  solver_type m_Solver;
  value_type_array m_TempRhs;

  void setSolutionMethod(const int *solverParams);
  void setSolverParameters(const int *solverParams);
};

} // namespace tacho