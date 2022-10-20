#include "TpetraLinearSolver.hpp"
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Ifpack2_Factory.hpp>
#include <MueLu.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include "PlatoUtilities.hpp"
#include <ios>
#include <limits>

namespace Plato {
/******************************************************************************//**
 * \brief get view from device
 *
 * \param[in] aView data on device
 * @returns Mirror on host
**********************************************************************************/
template <typename ViewType>
typename ViewType::HostMirror
get(const ViewType & aView)
{
    using RetType = typename ViewType::HostMirror;
    RetType tView = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView, aView);
    return tView;
}

/******************************************************************************//**
 * \brief Abstract system interface

   This class contains the node and dof map information and permits persistence
   of this information between solutions.
**********************************************************************************/
TpetraSystem::TpetraSystem(
    int            aNumNodes,
    Comm::Machine  aMachine,
    int            aDofsPerNode
) : mMatrixConversionTimer(Teuchos::TimeMonitor::getNewTimer("Analyze: Matrix Conversion")),
    mVectorConversionTimer(Teuchos::TimeMonitor::getNewTimer("Analyze: Vector Conversion"))
{
    mComm = aMachine.teuchosComm;

    int tNumDofs = aNumNodes*aDofsPerNode;

    mMap = Teuchos::rcp( new Tpetra_Map(tNumDofs, 0, mComm));

}

/******************************************************************************//**
 * \brief Convert from Plato::CrsMatrix<Plato::OrdinalType> to Tpetra_Matrix
**********************************************************************************/
Teuchos::RCP<Tpetra_Matrix>
TpetraSystem::fromMatrix(Plato::CrsMatrix<Plato::OrdinalType> aInMatrix) const
{
  Teuchos::TimeMonitor LocalTimer(*mMatrixConversionTimer);

  Plato::OrdinalType tMaxColSize = 0;
  {
    auto tRowMap = aInMatrix.rowMap();
    const Plato::OrdinalType tSize = tRowMap.size() - 1; // rowmap is num rows + 1

    Kokkos::Max<Plato::OrdinalType> tMaxReducer(tMaxColSize);
    Kokkos::parallel_reduce("KokkosReductionOperations::max", Kokkos::RangePolicy<>(0, tSize),
    KOKKOS_LAMBDA(const OrdinalType & aIndex, Plato::OrdinalType & aValue){
      tMaxReducer.join(aValue, tRowMap[aIndex+1] - tRowMap[aIndex]);
    }, tMaxReducer);
  }

  auto tRetVal = Teuchos::rcp(new Tpetra_Matrix(mMap, tMaxColSize*aInMatrix.numColsPerBlock()));

  auto tNumRowsPerBlock = aInMatrix.numRowsPerBlock();
  auto tNumColsPerBlock = aInMatrix.numColsPerBlock();
  auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

  std::vector<Plato::OrdinalType> tGlobalColumnIndices(tNumColsPerBlock);
  std::vector<Plato::Scalar>      tGlobalColumnValues (tNumColsPerBlock);

  auto tRowMap = get(aInMatrix.rowMap());
  auto tColMap = get(aInMatrix.columnIndices());
  auto tValues = get(aInMatrix.entries());

  auto tNumRows = tRowMap.extent(0)-1;
  size_t tCrsMatrixGlobalNumRows = tNumRows * tNumRowsPerBlock;
  size_t tTpetraGlobalNumRows = tRetVal->getGlobalNumRows();
  if(tCrsMatrixGlobalNumRows != tTpetraGlobalNumRows)
    throw std::domain_error("Input Plato::CrsMatrix size does not match TpetraSystem map.\n");

  for(Plato::OrdinalType iRowIndex=0; iRowIndex<tNumRows; iRowIndex++)
  {
      auto tFrom = tRowMap(iRowIndex);
      auto tTo   = tRowMap(iRowIndex+1);
      for(auto iColMapEntryIndex=tFrom; iColMapEntryIndex<tTo; iColMapEntryIndex++)
      {
          auto tBlockColIndex = tColMap(iColMapEntryIndex);
          for(Plato::OrdinalType iLocalRowIndex=0; iLocalRowIndex<tNumRowsPerBlock; iLocalRowIndex++)
          {
              auto tRowIndex = iRowIndex * tNumRowsPerBlock + iLocalRowIndex;
              for(Plato::OrdinalType iLocalColIndex=0; iLocalColIndex<tNumColsPerBlock; iLocalColIndex++)
              {
                  auto tColIndex = tBlockColIndex * tNumColsPerBlock + iLocalColIndex;
                  auto tSparseIndex = iColMapEntryIndex * tBlockSize + iLocalRowIndex * tNumColsPerBlock + iLocalColIndex;
                  tGlobalColumnIndices[iLocalColIndex] = tColIndex;
                  tGlobalColumnValues[iLocalColIndex]  = tValues[tSparseIndex];
              }
              Teuchos::ArrayView<const Plato::OrdinalType> tGlobalColumnIndicesView(tGlobalColumnIndices);
              Teuchos::ArrayView<const Plato::Scalar> tGlobalColumnValuesView(tGlobalColumnValues);
              tRetVal->insertGlobalValues(tRowIndex,tGlobalColumnIndicesView,tGlobalColumnValuesView);
          }
      }
  }

  tRetVal->fillComplete();

  return tRetVal;
}

/******************************************************************************//**
 * \brief Convert from ScalarVector to Tpetra_MultiVector
**********************************************************************************/
Teuchos::RCP<Tpetra_MultiVector>
TpetraSystem::fromVector(const Plato::ScalarVector tInVector) const
{
  Teuchos::TimeMonitor LocalTimer(*mVectorConversionTimer);
  auto tOutVector = Teuchos::rcp(new Tpetra_MultiVector(mMap, 1));
  if(tInVector.extent(0) != tOutVector->getLocalLength())
    throw std::domain_error("ScalarVector size does not match TpetraSystem map\n");

  auto tOutVectorDeviceView2D = tOutVector->getLocalView<Plato::DeviceType>(Tpetra::Access::ReadWrite);
  auto tOutVectorDeviceView1D = Kokkos::subview(tOutVectorDeviceView2D,Kokkos::ALL(), 0);

  Kokkos::deep_copy(tOutVectorDeviceView1D,tInVector);

  return tOutVector;
}

/******************************************************************************//**
 * \brief Convert from Tpetra_MultiVector to ScalarVector
**********************************************************************************/
void 
TpetraSystem::toVector(Plato::ScalarVector& tOutVector, const Teuchos::RCP<Tpetra_MultiVector> tInVector) const
{
    Teuchos::TimeMonitor LocalTimer(*mVectorConversionTimer);
    auto tLength = tInVector->getLocalLength();
    auto tTemp = Teuchos::rcp(new Tpetra_MultiVector(mMap, 1));
    if(tLength != tTemp->getLocalLength())
      throw std::domain_error("Tpetra_MultiVector map does not match TpetraSystem map.");

    if(tOutVector.extent(0) != tTemp->getLocalLength())
      throw std::range_error("ScalarVector does not match TpetraSystem map.");

    auto tInVectorDeviceView2D = tInVector->getLocalView<Plato::DeviceType>(Tpetra::Access::ReadWrite);
    auto tInVectorDeviceView1D = Kokkos::subview(tInVectorDeviceView2D,Kokkos::ALL(), 0);

    Kokkos::deep_copy(tOutVector,tInVectorDeviceView1D);
}

void
TpetraLinearSolver::initialize()
{

    setupSolverOptions();

    setupPreconditionerOptions();
}

TpetraLinearSolver::TpetraLinearSolver(
    const Teuchos::ParameterList&                   aSolverParams,
    int                                             aNumNodes,
    Comm::Machine                                   aMachine,
    int                                             aDofsPerNode,
    std::shared_ptr<Plato::MultipointConstraints>   aMPCs
) :
    AbstractSolver(aSolverParams, aMPCs),
    mSolverParams(aSolverParams),
    mSystem(Teuchos::rcp( new TpetraSystem(aNumNodes, aMachine, aDofsPerNode))),
    mPreLinearSolveTimer(Teuchos::TimeMonitor::getNewTimer("Analyze: Pre Linear Solve Setup")),
    mPreconditionerSetupTimer(Teuchos::TimeMonitor::getNewTimer("Analyze: Preconditioner Setup")),
    mLinearSolverTimer(Teuchos::TimeMonitor::getNewTimer("Analyze: Tpetra Linear Solve")),
    mSolverEndTime(mPreLinearSolveTimer->wallTime()),
    mDofsPerNode(aDofsPerNode)
{
    this->initialize();
}

template<class TpetraMatrixType>
Teuchos::RCP<Tpetra::Operator<typename TpetraMatrixType::scalar_type,
                              typename TpetraMatrixType::local_ordinal_type,
                              typename TpetraMatrixType::global_ordinal_type,
                              typename TpetraMatrixType::node_type> >
createIFpack2Preconditioner (const Teuchos::RCP<const TpetraMatrixType>& A,
                      const std::string& precondType,
                      const Teuchos::ParameterList& plist)
{
  typedef typename TpetraMatrixType::scalar_type scalar_type;
  typedef typename TpetraMatrixType::local_ordinal_type local_ordinal_type;
  typedef typename TpetraMatrixType::global_ordinal_type global_ordinal_type;
  typedef typename TpetraMatrixType::node_type node_type;

  typedef Ifpack2::Preconditioner<scalar_type, local_ordinal_type, 
                                  global_ordinal_type, node_type> prec_type;

  Teuchos::RCP<prec_type> prec;
  Ifpack2::Factory factory;
  prec = factory.create (precondType, A);
  prec->setParameters (plist);

  prec->initialize();
  prec->compute();

  return prec;
}

template<typename T>
inline void
TpetraLinearSolver::addDefaultToParameterList (Teuchos::ParameterList &aParams, const std::string &aEntryName, const T &aDefaultValue)
{
  if(!aParams.isType<T>(aEntryName))
    aParams.set(aEntryName, aDefaultValue);
}

void
TpetraLinearSolver::setupSolverOptions ()
{
  // mPreLinearSolveTimer->start();
  if(mSolverParams.isType<int>("Iterations"))
  {
      mNumIterations = mSolverParams.get<int>("Iterations");
  }
  else
  {
      mNumIterations = 100;
  }

  if(mSolverParams.isType<double>("Tolerance"))
  {
      mTolerance = mSolverParams.get<double>("Tolerance");
  }
  else
  {
      mTolerance = 1e-6;
  }

    mPreLinearSolveTimer->start();
  std::string tSolver = "";
  if (mSolverParams.isType<std::string>("Solver"))
    tSolver = mSolverParams.get<std::string>("Solver");
  else
    tSolver = "pseudoblock gmres";

  mSolver = Plato::tolower(tSolver);

  if (mSolver == "gmres")
  {
    mSolver = "pseudoblock gmres";
    REPORT("Tpetra using 'Pseudoblock GMRES' solver instead of user-specified 'GMRES' since matrix has block structure.")
  }
  else if (mSolver == "cg")
  {
    mSolver = "pseudoblock cg";
    REPORT("Tpetra using 'Pseudoblock CG' solver instead of user-specified 'CG' since matrix has block structure.")
  }

  mDisplayIterations = 0;
  if (mSolverParams.isType<int>("Display Iterations"))
    mDisplayIterations = mSolverParams.get<int>("Display Iterations");

  // Set default values here
  int tMaxIterations = 1000;
  double tTolerance  = 1e-8;
  if(mSolverParams.isType<int>("Iterations"))
    tMaxIterations = mSolverParams.get<int>("Iterations");
  if(mSolverParams.isType<double>("Tolerance"))
      tTolerance = mSolverParams.get<double>("Tolerance");

  if(mSolverParams.isType<Teuchos::ParameterList>("Solver Options"))
    mSolverOptions = mSolverParams.get<Teuchos::ParameterList>("Solver Options");

  this->addDefaultToParameterList(mSolverOptions, "Maximum Iterations",    tMaxIterations);
  this->addDefaultToParameterList(mSolverOptions, "Convergence Tolerance", tTolerance);
  this->addDefaultToParameterList(mSolverOptions, "Block Size",            mDofsPerNode);

  if (mSolver == "pseudoblock gmres")
    this->addDefaultToParameterList(mSolverOptions, "Num Blocks", tMaxIterations); // This is the number of iterations between restarts

  bool tPrintSolverParameterLists = false;
  if (mSolverParams.isType<bool>("Print Solver Parameters"))
    tPrintSolverParameterLists = mSolverParams.get<bool>("Print Solver Parameters");

  if (tPrintSolverParameterLists)
  {
    printf("\n'Linear Solver' Parameter List: \n");
    mSolverParams.print(std::cout, 2, true);
    printf("\n'Solver Options' sublist of 'Linear Solver' Parameter List: \n");
    mSolverOptions.print(std::cout, 2, true);
    printf("\n'Preconditioner Options' sublist of 'Linear Solver' Parameter List: \n");
    mPreconditionerOptions.print(std::cout, 2, true);
  }
}

void
TpetraLinearSolver::setupPreconditionerOptions ()
{
  std::string tPreconditionerPackage = "muelu";
  if (mSolverParams.isType<std::string>("Preconditioner Package"))
    tPreconditionerPackage = mSolverParams.get<std::string>("Preconditioner Package");
  mPreconditionerPackage = Plato::tolower(tPreconditionerPackage);

  mPreconditionerType = "Not Set";
  if (mSolverParams.isType<std::string>("Preconditioner Type"))
    mPreconditionerType = mSolverParams.get<std::string>("Preconditioner Type");
  else if (mPreconditionerPackage == "ifpack2")
    mPreconditionerType = "ILUT";

  if(mSolverParams.isType<Teuchos::ParameterList>("Preconditioner Options"))
    mPreconditionerOptions = mSolverParams.get<Teuchos::ParameterList>("Preconditioner Options");
  
  if (mPreconditionerPackage != "muelu") return;

  bool tUseSmoothedAggregation = true;
  if(mSolverParams.isType<bool>("Use Smoothed Aggregation"))
    tUseSmoothedAggregation = mSolverParams.get<bool>("Use Smoothed Aggregation");

  this->addDefaultToParameterList(mPreconditionerOptions, "number of equations", mDofsPerNode); // Same as 'Block Size' above in solver options
  this->addDefaultToParameterList(mPreconditionerOptions, "verbosity", std::string("none"));
  this->addDefaultToParameterList(mPreconditionerOptions, "coarse: max size", static_cast<int>(128));
  if (tUseSmoothedAggregation)
    this->addDefaultToParameterList(mPreconditionerOptions, "multigrid algorithm", std::string("sa"));
  else
    this->addDefaultToParameterList(mPreconditionerOptions, "multigrid algorithm", std::string("unsmoothed"));
  this->addDefaultToParameterList(mPreconditionerOptions, "transpose: use implicit", true);
  this->addDefaultToParameterList(mPreconditionerOptions, "max levels", static_cast<int>(10));
  this->addDefaultToParameterList(mPreconditionerOptions, "sa: use filtered matrix", true);
  this->addDefaultToParameterList(mPreconditionerOptions, "aggregation: type", std::string("uncoupled"));
  this->addDefaultToParameterList(mPreconditionerOptions, "aggregation: drop scheme", std::string("classical"));
  //this->addDefaultToParameterList(mPreconditionerOptions, "aggregation: drop tol", static_cast<double>(0.02));

  // Setup the smoother for the AMG preconditioner
  std::string tPreconditionerSmoother = "symmetric gs";
  if(mSolverParams.isType<std::string>("Preconditioner Smoother"))
    tPreconditionerSmoother = Plato::tolower(mSolverParams.get<std::string>("Preconditioner Smoother"));

  this->addDefaultToParameterList(mPreconditionerOptions, "smoother: type", std::string("RELAXATION"));
  Teuchos::ParameterList & tSmootherParams = mPreconditionerOptions.sublist("smoother: params");
  if (tPreconditionerSmoother == "symmetric gs")
    this->addDefaultToParameterList(tSmootherParams, "relaxation: type", std::string("MT Symmetric Gauss-Seidel"));
  else if (tPreconditionerSmoother == "gs")
    this->addDefaultToParameterList(tSmootherParams, "relaxation: type", std::string("MT Gauss-Seidel"));
  else
    this->addDefaultToParameterList(tSmootherParams, "relaxation: type", std::string("Jacobi"));
  this->addDefaultToParameterList(tSmootherParams, "relaxation: sweeps", static_cast<int>(2));
  this->addDefaultToParameterList(tSmootherParams, "relaxation: damping factor", static_cast<double>(0.9));

}

template<class MV, class OP>
void
TpetraLinearSolver::belosSolve (Teuchos::RCP<const OP> A, Teuchos::RCP<MV> X, Teuchos::RCP<const MV> B, Teuchos::RCP<const OP> M) 
{
  Teuchos::TimeMonitor LocalTimer(*mLinearSolverTimer);

  using scalar_type = typename MV::scalar_type;
  Teuchos::RCP<Teuchos::ParameterList> tSolverOptions = Teuchos::rcp(new Teuchos::ParameterList(mSolverOptions));
  Belos::SolverFactory<scalar_type, MV, OP> factory;
  Teuchos::RCP<Belos::SolverManager<scalar_type, MV, OP> > solver = factory.create (mSolver, tSolverOptions);

  typedef Belos::LinearProblem<scalar_type, MV, OP> problem_type;
  Teuchos::RCP<problem_type> problem = Teuchos::rcp (new problem_type(A, X, B));

  problem->setRightPrec(M);
  
  problem->setProblem();
  solver->setProblem (problem);

  Belos::ReturnType result = solver->solve();
  mNumIterations           = solver->getNumIters();
  mAchievedTolerance       = solver->achievedTol();

  if (result == Belos::Unconverged) {
    Plato::Scalar tTolerance = static_cast<Plato::Scalar>(100.0) * std::numeric_limits<Plato::Scalar>::epsilon();
    if (mAchievedTolerance > tTolerance) {
        std::stringstream errorMessage;
        errorMessage << "Tpetra Warning: Belos solver did not achieve desired tolerance." <<
                        "Completed " << mNumIterations << " iterations, achieved absolute tolerance of " <<
                        std::scientific << mAchievedTolerance << " (not relative)" << std::endl;
        ANALYZE_THROWERR(errorMessage.str());
    }
  }
}

/******************************************************************************//**
 * \brief Solve the linear system
**********************************************************************************/
void
TpetraLinearSolver::innerSolve(
    Plato::CrsMatrix<Plato::OrdinalType> aA,
    Plato::ScalarVector   aX,
    Plato::ScalarVector   aB
)
{
  mPreLinearSolveTimer->stop(); 
  mPreLinearSolveTimer->incrementNumCalls();
  mSolverStartTime = mPreLinearSolveTimer->wallTime();
  const double tAnalyzeElapsedTime = mSolverStartTime - mSolverEndTime;

  Teuchos::RCP<Tpetra_Matrix> A = mSystem->fromMatrix(aA);
  Teuchos::RCP<Tpetra_MultiVector> X = mSystem->fromVector(aX);
  Teuchos::RCP<Tpetra_MultiVector> B = mSystem->fromVector(aB);

  Teuchos::RCP<Tpetra_Operator> M;

  mPreconditionerSetupTimer->start();
  if(mPreconditionerPackage == "ifpack2")
    M = createIFpack2Preconditioner<Tpetra_Matrix> (A, mPreconditionerType, mPreconditionerOptions);
  else if(mPreconditionerPackage == "muelu")
    M = MueLu::CreateTpetraPreconditioner(static_cast<Teuchos::RCP<Tpetra_Operator>>(A), mPreconditionerOptions);
  else
  {
    std::string tInvalid_solver = "Preconditioner Package " + mPreconditionerPackage 
                                + " is not currently a valid option. Valid options: ('ifpack2', 'muelu')\n";
    throw std::invalid_argument(tInvalid_solver);
  }
  mPreconditionerSetupTimer->stop();
  mPreconditionerSetupTimer->incrementNumCalls(); 

  belosSolve<Tpetra_MultiVector, Tpetra_Operator> (A, X, B, M);

  mSystem->toVector(aX,X);

  mSolverEndTime = mPreLinearSolveTimer->wallTime();
  const double tTpetraElapsedTime = mSolverEndTime - mSolverStartTime;
  if (mDisplayIterations > 0)
    printf("Pre Lin. Solve %5.1f second(s) || Tpetra Lin. Solve %5.1f second(s), %4d iteration(s), %7.1e achieved tolerance\n",
           tAnalyzeElapsedTime, tTpetraElapsedTime, mNumIterations, mAchievedTolerance);
  mPreLinearSolveTimer->start();
}

} // end namespace Plato
