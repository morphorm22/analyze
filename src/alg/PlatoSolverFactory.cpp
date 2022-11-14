#include "alg/PlatoSolverFactory.hpp"
#include "PlatoUtilities.hpp"

#include "alg/AmgXLinearSolver.hpp"
#include "alg/EpetraLinearSolver.hpp"
#ifdef PLATO_TPETRA
#include "alg/TpetraLinearSolver.hpp"
#endif
#ifdef PLATO_TACHO
#include "alg/TachoLinearSolver.hpp"
#endif

namespace Plato {

std::string determine_solver_stack(const Teuchos::ParameterList& tSolverParams)
{
  std::string tSolverStack;
  if(tSolverParams.isType<std::string>("Solver Stack"))
  {
      tSolverStack = tSolverParams.get<std::string>("Solver Stack");
  }
  else
  {
#ifdef PLATO_TACHO
      tSolverStack = "Tacho";
#elif HAVE_AMGX
      tSolverStack = "AmgX";
#elif PLATO_TPETRA
      tSolverStack = "Tpetra";
#elif PLATO_EPETRA
      tSolverStack = "Epetra";
#else
      ANALYZE_THROWERR("PLato Analyze was compiled without a linear solver!.  Exiting.");
#endif
  }

  return tSolverStack;
}

/******************************************************************************//**
 * @brief Solver factory for AbstractSolvers with MPCs
**********************************************************************************/
rcp<AbstractSolver>
SolverFactory::create(
    Plato::OrdinalType                              aNumNodes,
    Comm::Machine                                   aMachine,
    Plato::OrdinalType                              aDofsPerNode,
    std::shared_ptr<Plato::MultipointConstraints>   aMPCs
)
{
  auto tSolverStack = Plato::determine_solver_stack(mSolverParams);
  auto tLowerSolverStack = Plato::tolower(tSolverStack);

  if(tLowerSolverStack == "epetra")
  {
#ifdef PLATO_EPETRA
      const Plato::OrdinalType tNumCondensedNodes = (aMPCs == nullptr) ? aNumNodes : aMPCs->getNumCondensedNodes();
      return std::make_shared<Plato::EpetraLinearSolver>(mSolverParams, tNumCondensedNodes, aMachine, aDofsPerNode, aMPCs);
#else
      ANALYZE_THROWERR("Not compiled with Epetra");
#endif
  }
  else if(tLowerSolverStack == "tpetra")
  {
#ifdef PLATO_TPETRA
      const Plato::OrdinalType tNumCondensedNodes = (aMPCs == nullptr) ? aNumNodes : aMPCs->getNumCondensedNodes();
      return std::make_shared<Plato::TpetraLinearSolver>(mSolverParams, tNumCondensedNodes, aMachine, aDofsPerNode, aMPCs);
#else
      ANALYZE_THROWERR("Not compiled with Tpetra");
#endif
  }
  else if(tLowerSolverStack == "amgx")
  {
#ifdef HAVE_AMGX
      return std::make_shared<Plato::AmgXLinearSolver>(mSolverParams, aDofsPerNode, aMPCs);
#else
      ANALYZE_THROWERR("Not compiled with AmgX");
#endif
  }
  else if(tLowerSolverStack == "tacho")
  {
#ifdef PLATO_TACHO
      return std::make_shared<tacho::TachoLinearSolver>(mSolverParams, mType, aMPCs);
#else
      ANALYZE_THROWERR("Not compiled with Tacho");
#endif
  }
  ANALYZE_THROWERR("Requested solver stack not found");
}

} // end namespace Plato
