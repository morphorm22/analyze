#ifndef PLATO_EPETRA_SOLVER_HPP
#define PLATO_EPETRA_SOLVER_HPP

#include "PlatoAbstractSolver.hpp"
#include "alg/ParallelComm.hpp"

#include <AztecOO.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_MpiComm.h>
#include <Epetra_VbrMatrix.h>
#include <Epetra_VbrRowMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>


namespace Plato {

/******************************************************************************//**
 * \brief Abstract system interface

   This class contains the node and dof map information and permits persistence
   of this information between solutions.
**********************************************************************************/
class EpetraSystem
{
    rcp<Epetra_BlockMap> mBlockRowMap;
    rcp<Epetra_Comm>     mComm;

    Teuchos::RCP<Teuchos::Time> mMatrixConversionTimer;
    Teuchos::RCP<Teuchos::Time> mVectorConversionTimer;

  public:
    EpetraSystem(
        int            aNumNodes,
        Comm::Machine  aMachine,
        int            aDofsPerNode
    );

    /******************************************************************************//**
     * \brief Convert from Plato::CrsMatrix<Plato::OrdinalType> to Epetra_VbrMatrix
    **********************************************************************************/
    rcp<Epetra_VbrMatrix>
    fromMatrix(Plato::CrsMatrix<Plato::OrdinalType> tInMatrix) const;

    /******************************************************************************//**
     * \brief Convert from ScalarVector to Epetra_Vector
    **********************************************************************************/
    rcp<Epetra_Vector>
    fromVector(Plato::ScalarVector tInVector) const;

    /******************************************************************************//**
     * \brief Convert from Epetra_Vector to ScalarVector
    **********************************************************************************/
    void
    toVector(Plato::ScalarVector tOutVector, rcp<Epetra_Vector> tInVector) const;

    /******************************************************************************//**
     * \brief get EpetraSystem map 
    **********************************************************************************/
    rcp<Epetra_BlockMap> getMap() const {return mBlockRowMap;}
};

/******************************************************************************//**
 * \brief Concrete EpetraLinearSolver
**********************************************************************************/
class EpetraLinearSolver : public AbstractSolver
{
    rcp<EpetraSystem> mSystem;

    Teuchos::ParameterList mSolverParams;

    Teuchos::RCP<Teuchos::Time> mLinearSolverTimer;

    int mIterations = 1000; /*!< maximum linear solver iterations */
    int mDisplayIterations = 0; /*!< display solver iterations history to console */
    Plato::Scalar mTolerance = 1e-10; /*!< linear solver tolerance */

  public:
    EpetraLinearSolver(
        const Teuchos::ParameterList&                   aSolverParams,
        int                                             aNumNodes,
        Comm::Machine                                   aMachine,
        int                                             aDofsPerNode,
        std::shared_ptr<Plato::MultipointConstraints>   aMPCs = nullptr
    );

    /******************************************************************************//**
     * \brief Solve the linear system
    **********************************************************************************/
    void
    innerSolve(
        Plato::CrsMatrix<Plato::OrdinalType> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) override;

    /******************************************************************************//**
     * \brief Setup the AztecOO solver
    **********************************************************************************/
    void
    setupSolver(AztecOO& aSolver);
};

} // end namespace Plato

#endif
