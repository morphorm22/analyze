#ifndef PLATO_SOLVER_FACTORY_HPP
#define PLATO_SOLVER_FACTORY_HPP

#include "alg/AmgXLinearSolver.hpp"
#ifdef PLATO_EPETRA
#include "alg/EpetraLinearSolver.hpp"
#endif
#ifdef PLATO_TPETRA
#include "alg/TpetraLinearSolver.hpp"
#endif

namespace Plato {

/******************************************************************************//**
 * \brief Solver factory for AbstractSolvers
**********************************************************************************/
class SolverFactory
{
    const Teuchos::ParameterList& mSolverParams;

  public:
    SolverFactory(
        Teuchos::ParameterList& aSolverParams
    ) : mSolverParams(aSolverParams) { }

    rcp<AbstractSolver>
    create(
        Plato::OrdinalType                              aNumNodes,
        Comm::Machine                                   aMachine,
        Plato::OrdinalType                              aDofsPerNode
    );

    rcp<AbstractSolver>
    create(
        Plato::OrdinalType                              aNumNodes,
        Comm::Machine                                   aMachine,
        Plato::OrdinalType                              aDofsPerNode,
        std::shared_ptr<Plato::MultipointConstraints>   aMPCs
    );
};

} // end Plato namespace

#endif
