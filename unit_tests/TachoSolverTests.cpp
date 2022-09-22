#include <algorithm>
#include <array>
#include <math.h>

#include "alg/TachoLinearSolver.hpp"

#include "Teuchos_UnitTestHarness.hpp"

namespace tachoSolverTest {
namespace {
template <class SX>
double solutionError(const int numRows,
		     const SX* rhs, 
		     const SX* sol, 
		     const int* rowBegin, 
		     const int* columns, 
		     const double* values)
{
  double normRhsSquared(0), normErrorSquared(0);
  for (int i=0; i<numRows; i++) {
    SX resid = rhs[i];
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      resid -= values[j]*sol[col];
    }
    double absVal = std::abs(rhs[i]);
    normRhsSquared += absVal*absVal;
    absVal = std::abs(resid);
    normErrorSquared += absVal*absVal;
  }
  double normError = std::sqrt(normErrorSquared);
  double normRhs = std::sqrt(normRhsSquared);
  return normError/normRhs;
}

template <class SX>
void makeNonSymmetric(const int numRows, 
                      const int* rowBegin, 
                      const int* columns, 
                      SX* values)
{
  for (int i=0; i<numRows; i++) {
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      if (col > i) values[j] *= 1.01;
    }
  }
}

template <class SX>
void sortColumns(const int numRows, 
                 const int* rowBegin, 
                 int* columns, 
                 SX* values)
{
  std::vector<std::pair<int,int>> vals(numRows);
  std::vector<SX> valuesCopy(numRows);
  for (int i=0; i<numRows; i++) {
    const int numCols = rowBegin[i+1] - rowBegin[i];
    for (int j=0; j<numCols; j++) {
      const int index = rowBegin[i] + j;
      vals[j] = std::make_pair(columns[index], j);
      valuesCopy[j] = values[index];
    }
    std::sort(vals.begin(), vals.begin()+numCols);
    for (int j=0; j<numCols; j++) {
      const int index = rowBegin[i] +j;
      columns[index] = vals[j].first;
      values[index] = valuesCopy[vals[j].second];
    }
  }
}

template <class SX>
bool solveLinearSystems(tacho::tachoSolver<SX> & solver, 
                        Kokkos::Timer & timer,
                        const int numRows,
                        int* rowBegin,
                        int* columns,
                        SX* values)
{
  const double tol=1e-10;
  //  std vector right hand side
  //  if an application uses std vector for interfacing rhs,
  //  it requires additional copy. it is better to directly 
  //  use a kokkos device view.
  std::vector<SX> rhs(numRows), sol(numRows);
  { // randomize rhs
    const unsigned int seed = 0;
    srand(seed);
    for (int i=0;i<numRows;++i)
      rhs[i] = rand()/((double)RAND_MAX+1.0);
  }

  //    this example only works for single right hand side
  const int NRHS = 1;
  typedef Kokkos::View<SX**, Kokkos::LayoutLeft, tacho::device_type> ViewVectorType;
  ViewVectorType x("x", numRows, NRHS);

#if defined (KOKKOS_ENABLE_CUDA)
  //  transfer b into device
  ViewVectorType b(Kokkos::ViewAllocateWithoutInitializing("b"), numRows, NRHS);
  Kokkos::deep_copy(Kokkos::subview(b, Kokkos::ALL(), 0), 
                    Kokkos::View<SX*,tacho::device_type>(rhs.data(), numRows));
#else
  //  wrap rhs data with view
  ViewVectorType b(rhs.data(), numRows, NRHS);
#endif

  timer.reset();
  
  const int numRuns = 10;
  for (int run=0; run<numRuns; run++) {
    solver.MySolve(NRHS, b, x);
    Kokkos::fence();
  }
  double runTime = timer.seconds();

  //  computing residual on host
  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);
  
  std::cout << "tacho:: solve time (" << numRuns << ") = " << runTime << std::endl;
  double relativeError = solutionError(numRows, rhs.data(), h_x.data(), rowBegin, columns, values);
  std::cout << "tacho:: relative error = " << relativeError << std::endl;
  if (relativeError > tol) {
    std::cout << "tacho FAILS (rel Error = " 
              << relativeError << ", tol = " << tol << std::endl;
  } else {
    std::cout << "tacho PASS " << std::endl;
  }

  return relativeError < tol;
}

template <class SX>
bool testTachoSolver(int numRows,
		     int* rowBegin,
		     int* columns,
		     SX* values,
             const int sym)
{
  Kokkos::Timer timer;
  //  Tacho options
  int solutionMethod = 1; // Cholesky
  if (sym == 1) {
    makeNonSymmetric(numRows, rowBegin, columns, values);
    sortColumns(numRows, rowBegin, columns, values);
    solutionMethod = 3; // SymLU
  }
  std::vector<int> tachoParams;
  tacho::getTachoParams(tachoParams, solutionMethod);
  tachoParams[tacho::VERBOSITY] = 1;
  //  Tacho solver analyze + factorize (fence is required)
  tacho::tachoSolver<SX> solver(tachoParams.data());
  const bool printTimings = true;
  solver.Initialize(numRows, rowBegin, columns, values, printTimings);
  Kokkos::fence();

  bool success;

  success = solveLinearSystems(solver, timer, numRows, rowBegin, columns, values);
  // Next, update solver factorization for a perturbed matrix
  for (int i=0; i<numRows; i++) {
    for (int j=rowBegin[i]; j<rowBegin[i+1]; j++) {
      const int col = columns[j];
      if (col == i) values[j] *= 1.01;
    }
  }
  const int numTerms = rowBegin[numRows];
  solver.refactorMatrix(numTerms, values);
  std::cout << "results after refactoring (numeric only) matrix" << std::endl;
  success = success && solveLinearSystems(solver, timer, numRows, rowBegin, columns, values);

  return success;
}
}

TEUCHOS_UNIT_TEST(tachoSolver, Test1)
{
  int numRows = 4;
  int rowBegin[] = {0, 2, 5, 8, 10}; 
  int columns[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
  double values[] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};

  std::cout << "number of rows = " << numRows << std::endl;

  std::cout << "testing double type (symmetric matrix)------------------\n";
  int sym = 2; // symmetric positive definite matrix
  TEST_ASSERT(testTachoSolver<double>(numRows, rowBegin, columns, values, sym));

  std::cout << "testing double type (non-symmetric matrix)--------------\n";
  sym = 1; // nonsymmetric, but structurally symmetric, matrix
  TEST_ASSERT(testTachoSolver<double>(numRows, rowBegin, columns, values, sym));
}

} // end namespace

