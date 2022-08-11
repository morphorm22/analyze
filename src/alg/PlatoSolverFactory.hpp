#pragma once

#include "Teuchos_ParameterList.hpp"
#include "PlatoAbstractSolver.hpp"
#include "alg/ParallelComm.hpp"

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
        Plato::OrdinalType                              aDofsPerNode,
        std::shared_ptr<Plato::MultipointConstraints>   aMPCs = nullptr
    );
};

} // end Plato namespace
