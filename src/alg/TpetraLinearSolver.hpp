#ifndef PLATO_TPETRA_SOLVER_HPP
#define PLATO_TPETRA_SOLVER_HPP

#include "PlatoAbstractSolver.hpp"
#include "alg/ParallelComm.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace Plato {

  using Tpetra_Map = Tpetra::Map<int, Plato::OrdinalType>;
  using Tpetra_MultiVector = Tpetra::MultiVector<Plato::Scalar, int, Plato::OrdinalType>;
  using Tpetra_Vector = Tpetra::Vector<Plato::Scalar, int, Plato::OrdinalType>;
  using Tpetra_Matrix = Tpetra::CrsMatrix<Plato::Scalar, int, Plato::OrdinalType>;
  using Tpetra_Operator = Tpetra::Operator<Plato::Scalar, int, Plato::OrdinalType>;

/******************************************************************************//**
 * \brief Abstract system interface

   This class contains the node and dof map information and permits persistence
   of this information between solutions.
**********************************************************************************/
class TpetraSystem
{
  Teuchos::RCP<Tpetra_Map> mMap;
  Teuchos::RCP<const Teuchos::Comm<int>>  mComm;

  Teuchos::RCP<Teuchos::Time> mMatrixConversionTimer;
  Teuchos::RCP<Teuchos::Time> mVectorConversionTimer;

  public:
    TpetraSystem(
        int            aNumNodes,
        Comm::Machine  aMachine,
        int            aDofsPerNode
    );

    /******************************************************************************//**
     * \brief Convert from Plato::CrsMatrix<int> to Tpetra_Matrix
    **********************************************************************************/
    Teuchos::RCP<Tpetra_Matrix>
    fromMatrix(const Plato::CrsMatrix<Plato::OrdinalType> tInMatrix) const;

    /******************************************************************************//**
     * \brief Convert from ScalarVector to Tpetra_MultiVector
    **********************************************************************************/
    Teuchos::RCP<Tpetra_MultiVector>
    fromVector(const Plato::ScalarVector tInVector) const;

    /******************************************************************************//**
     * \brief Convert from Tpetra_MultiVector to ScalarVector
    **********************************************************************************/
    void
    toVector(Plato::ScalarVector& tOutVector, const Teuchos::RCP<Tpetra_MultiVector> tInVector) const;

    /******************************************************************************//**
     * \brief get TpetraSystem map 
    **********************************************************************************/
    Teuchos::RCP<Tpetra_Map> getMap() const {return mMap;}

  private:
      void checkInputMatrixSize(const Plato::CrsMatrix<Plato::OrdinalType> aInMatrix,
               Kokkos::View<Plato::OrdinalType*, MemSpace>::HostMirror aRowMap) const;
};

/******************************************************************************//**
 * \brief Concrete TpetraLinearSolver
**********************************************************************************/
class TpetraLinearSolver : public AbstractSolver
{
    Teuchos::RCP<TpetraSystem> mSystem;

    std::string mSolverPackage;
    std::string mSolver;
    std::string mPreconditionerPackage;
    std::string mPreconditionerType;

    Teuchos::ParameterList mSolverOptions;
    Teuchos::ParameterList mPreconditionerOptions;

    Teuchos::RCP<Teuchos::Time> mPreLinearSolveTimer;
    Teuchos::RCP<Teuchos::Time> mPreconditionerSetupTimer;
    Teuchos::RCP<Teuchos::Time> mLinearSolverTimer;
    double mSolverStartTime, mSolverEndTime;

    int    mDisplayIterations;
    int    mDofsPerNode;
    double mAchievedTolerance;

    Teuchos::ParameterList mSolverParams;

    int mNumIterations = 1000; /*!< maximum linear solver iterations */
    Plato::Scalar mTolerance = 1e-14; /*!< linear solver tolerance */
    
  public:
    TpetraLinearSolver(
        const Teuchos::ParameterList&                   aSolverParams,
        int                                             aNumNodes,
        Comm::Machine                                   aMachine,
        int                                             aDofsPerNode,
        std::shared_ptr<Plato::MultipointConstraints>   aMPCs = nullptr
    );

    void initialize();

    /******************************************************************************//**
     * @brief Solve the linear system
    **********************************************************************************/
    void
    innerSolve(
        Plato::CrsMatrix<Plato::OrdinalType> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) override;

  private:
    /******************************************************************************//**
     * \brief Setup the Belos solver and solve
    **********************************************************************************/
    template<class MV, class OP>
    void
    belosSolve (Teuchos::RCP<const OP> A, Teuchos::RCP<MV> X, Teuchos::RCP<const MV> B, Teuchos::RCP<const OP> M);

    /******************************************************************************//**
     * @brief Setup the solver options
    ********************************************************************* ************/
    void
    setupSolverOptions();

    /******************************************************************************//**
     * @brief Setup the preconditioner options
    ********************************************************************* ************/
    void
    setupPreconditionerOptions();

    /******************************************************************************//**
     * @brief Add to parameter list if not set by user
    ********************************************************************* ************/
    template<typename T>
    inline void
    addDefaultToParameterList (Teuchos::ParameterList &aParams, const std::string &aEntryName, const T &aDefaultValue);
};

} // end namespace Plato

#endif
