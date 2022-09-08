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
  public:
    SolverFactory(
        Teuchos::ParameterList& aSolverParams,
        LinearSystemType type = LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE
    ) : mSolverParams(aSolverParams), mType(type) { }

    rcp<AbstractSolver>
    create(
        Plato::OrdinalType                              aNumNodes,
        Comm::Machine                                   aMachine,
        Plato::OrdinalType                              aDofsPerNode,
        std::shared_ptr<Plato::MultipointConstraints>   aMPCs = nullptr
    );
private:
    const Teuchos::ParameterList& mSolverParams;
    const LinearSystemType mType;
};

} // end Plato namespace
