#include "TachoLinearSolver.hpp"

namespace tacho {

void getTachoParams(std::vector<int> &tachoParams, const int solutionMethod) {
  tachoParams.assign(tacho::INDEX_LENGTH, 0);
  tachoParams[tacho::SOLUTION_METHOD] = solutionMethod;
  tachoParams[tacho::USEDEFAULTSOLVERPARAMETERS] = 0;
  tachoParams[tacho::VERBOSITY] = 0;
  tachoParams[tacho::SMALLPROBLEMTHRESHOLDSIZE] = 1024;

#if defined(KOKKOS_ENABLE_CUDA)
  tachoParams[tacho::TASKING_OPTION_MAXNUMSUPERBLOCKS] = 32;
  tachoParams[tacho::TASKING_OPTION_BLOCKSIZE] = 64;
  tachoParams[tacho::TASKING_OPTION_PANELSIZE] = 32;

  tachoParams[tacho::LEVELSET_OPTION_SCHEDULING] = 1;
  tachoParams[tacho::LEVELSET_OPTION_DEVICE_LEVEL_CUT] = 0;
  tachoParams[tacho::LEVELSET_OPTION_DEVICE_FACTOR_THRES] = 64;
  tachoParams[tacho::LEVELSET_OPTION_DEVICE_SOLVE_THRES] = 128;
  tachoParams[tacho::LEVELSET_OPTION_NSTREAMS] = 8;
#else
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  tachoParams[tacho::TASKING_OPTION_MAXNUMSUPERBLOCKS] =
      std::max(tacho::host_space::thread_pool_size(0) / 2, 1);
#else
  tachoParams[tacho::TASKING_OPTION_MAXNUMSUPERBLOCKS] =
      std::max(tacho::host_space::impl_thread_pool_size(0) / 2, 1);
#endif
  tachoParams[tacho::TASKING_OPTION_BLOCKSIZE] = 256;
  tachoParams[tacho::TASKING_OPTION_PANELSIZE] = 128;

  tachoParams[tacho::LEVELSET_OPTION_SCHEDULING] = 0;
  // the following options are not used and set dummy values
  tachoParams[tacho::LEVELSET_OPTION_DEVICE_LEVEL_CUT] = 0;
  tachoParams[tacho::LEVELSET_OPTION_DEVICE_FACTOR_THRES] = 0;
  tachoParams[tacho::LEVELSET_OPTION_DEVICE_SOLVE_THRES] = 0;
  tachoParams[tacho::LEVELSET_OPTION_NSTREAMS] = 0;
#endif
}

template <typename SX>
tachoSolver<SX>::tachoSolver(const int *solverParams)
    : m_numRows(0), m_Solver() {
  setSolverParameters(solverParams);
}

template <typename SX> tachoSolver<SX>::~tachoSolver() { m_Solver.release(); }

template <typename SX>
void tachoSolver<SX>::refactorMatrix(const int numTerms, SX *values) {
#if defined(KOKKOS_ENABLE_CUDA)
  // transfer A into device
  value_type_array ax(Kokkos::ViewAllocateWithoutInitializing("ax"), numTerms);
  value_type_array_host ax_host(values, numTerms);
  Kokkos::deep_copy(ax, ax_host);
#else
  /// wrap pointer on host
  value_type_array ax(values, numTerms);
#endif
  m_Solver.factorize(ax);
  Kokkos::fence();
}

template <typename SX>
int tachoSolver<SX>::Initialize(
    int numRows,
    /// with TACHO_ENABLE_INT_INT, size_type is "int"
    int *rowBegin, int *columns, SX *values, const bool printTimings)

{
  m_numRows = numRows;
  if (m_numRows == 0)
    return 0;
  m_Solver.setOrderConnectedGraphSeparately(0);

  const int numTerms = rowBegin[numRows];
  size_type_array_host ap_host((size_type *)rowBegin, numRows + 1);
  ordinal_type_array_host aj_host((ordinal_type *)columns, numTerms);

#if defined(KOKKOS_ENABLE_CUDA)
  /// transfer A into device
  value_type_array ax(Kokkos::ViewAllocateWithoutInitializing("ax"), numTerms);
  value_type_array_host ax_host(values, numTerms);
  Kokkos::deep_copy(ax, ax_host);
#else
  /// wrap pointer on host
  value_type_array ax(values, numTerms);
#endif

  Kokkos::Timer timer;
  {
    if (printTimings)
      timer.reset();
    m_Solver.analyze(numRows, ap_host, aj_host);
    if (printTimings) {
      const double t = timer.seconds();
      std::cout << "ExternalInterface:: analyze time " << t << std::endl;
    }
  }
  {
    if (printTimings)
      timer.reset();
    m_Solver.initialize();
    if (printTimings) {
      const double t = timer.seconds();
      std::cout << "ExternalInterface:: initialize time " << t << std::endl;
    }
  }
  {
    if (printTimings)
      timer.reset();
    m_Solver.factorize(ax);
    if (printTimings) {
      const double t = timer.seconds();
      std::cout << "ExternalInterface:: factorize time " << t << std::endl;
    }
  }
  Kokkos::fence();

  return 0;
}

template <typename SX>
void tachoSolver<SX>::MySolve(int NRHS, value_type_matrix &b,
                              value_type_matrix &x) {
  if (m_numRows == 0)
    return;

  /// kyungjoo: lazy allocation of workspace
  if (NRHS != int(b.extent(1)) || NRHS != int(x.extent(1)))
    printf("Error: NRHS does not match to b and x extent(1)\n");
  { // this workspace is allocated when NRHS is changed
    const int rhs_span = m_numRows * NRHS, rhs_span_actual = m_TempRhs.span();
    if (rhs_span != rhs_span_actual) {
      m_TempRhs = value_type_array(
          Kokkos::ViewAllocateWithoutInitializing("m_TempRhs"), rhs_span);
    }
  }
  const auto t = value_type_matrix(m_TempRhs.data(), b.extent(0), b.extent(1));
  m_Solver.solve(x, b, t);
  Kokkos::fence();
}

template <typename SX>
void tachoSolver<SX>::setSolutionMethod(const int *solverParams) {
  // solution method (e.g. Cholesky, LDL, or SymLU)
  m_Solver.setSolutionMethod(solverParams[SOLUTION_METHOD]);
}

template <typename SX>
void tachoSolver<SX>::setSolverParameters(const int *solverParams) {
  if (solverParams[USEDEFAULTSOLVERPARAMETERS])
    return;

  // common options
  m_Solver.setVerbose(solverParams[VERBOSITY]);
  m_Solver.setSmallProblemThresholdsize(
      solverParams[SMALLPROBLEMTHRESHOLDSIZE]);

  // solution method (e.g. Cholesky, LDL, or SymLU)
  setSolutionMethod(solverParams);

  // tasking options
  m_Solver.setBlocksize(solverParams[TASKING_OPTION_BLOCKSIZE]);
  m_Solver.setPanelsize(solverParams[TASKING_OPTION_PANELSIZE]);
  m_Solver.setMaxNumberOfSuperblocks(
      solverParams[TASKING_OPTION_MAXNUMSUPERBLOCKS]);

  // levelset options
  m_Solver.setLevelSetScheduling(solverParams[LEVELSET_OPTION_SCHEDULING]);
  m_Solver.setLevelSetOptionDeviceLevelCut(
      solverParams[LEVELSET_OPTION_DEVICE_LEVEL_CUT]);
  m_Solver.setLevelSetOptionDeviceFunctionThreshold(
      solverParams[LEVELSET_OPTION_DEVICE_FACTOR_THRES],
      solverParams[LEVELSET_OPTION_DEVICE_SOLVE_THRES]);
  m_Solver.setLevelSetOptionNumStreams(solverParams[LEVELSET_OPTION_NSTREAMS]);
}

template class tachoSolver<double>;

}