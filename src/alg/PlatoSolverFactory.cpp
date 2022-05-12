#include "alg/PlatoSolverFactory.hpp"
#include "PlatoUtilities.hpp"

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
#ifdef HAVE_AMGX
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
 * \brief Solver factory for AbstractSolvers
**********************************************************************************/
rcp<AbstractSolver>
SolverFactory::create(
    Plato::OrdinalType                              aNumNodes,
    Comm::Machine                                   aMachine,
    Plato::OrdinalType                              aDofsPerNode
)
{
  auto tSolverStack = Plato::determine_solver_stack(mSolverParams);
  auto tLowerSolverStack = Plato::tolower(tSolverStack);

  if(tLowerSolverStack == "epetra")
  {
#ifdef PLATO_EPETRA
      return std::make_shared<Plato::EpetraLinearSolver>(mSolverParams, aNumNodes, aMachine, aDofsPerNode);
#else
      ANALYZE_THROWERR("Not compiled with Epetra");
#endif

  }
  else if(tLowerSolverStack == "tpetra")
  {
#ifdef PLATO_TPETRA
      return std::make_shared<Plato::TpetraLinearSolver>(mSolverParams, aNumNodes, aMachine, aDofsPerNode);
#else
      ANALYZE_THROWERR("Not compiled with Tpetra");
#endif
  }
  else if(tLowerSolverStack == "amgx")
  {
#ifdef HAVE_AMGX
      return std::make_shared<Plato::AmgXLinearSolver>(mSolverParams, aDofsPerNode);
#else
      ANALYZE_THROWERR("Not compiled with AmgX");
#endif
  }
  ANALYZE_THROWERR("Requested solver stack not found");
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
      Plato::OrdinalType tNumCondensedNodes = aMPCs->getNumCondensedNodes();
      return std::make_shared<Plato::EpetraLinearSolver>(mSolverParams, tNumCondensedNodes, aMachine, aDofsPerNode, aMPCs);
#else
      ANALYZE_THROWERR("Not compiled with Epetra");
#endif
  }
  else if(tLowerSolverStack == "tpetra")
  {
#ifdef PLATO_TPETRA
      Plato::OrdinalType tNumCondensedNodes = aMPCs->getNumCondensedNodes();
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
  ANALYZE_THROWERR("Requested solver stack not found");
}

} // end namespace Plato
