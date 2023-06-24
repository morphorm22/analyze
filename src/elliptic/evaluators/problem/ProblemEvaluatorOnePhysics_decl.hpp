/*
 * ProblemEvaluatorOnePhysics_decl.hpp
 *
 *  Created on: June 21, 2023
 */

#pragma once

#include "PlatoMesh.hpp"
#include "SpatialModel.hpp"
#include "base/Database.hpp"
#include "base/ProblemEvaluatorBase.hpp"

#include "solver/ParallelComm.hpp"
#include "solver/PlatoAbstractSolver.hpp"

#include "elliptic/base/VectorFunction.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorBase.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class ProblemEvaluatorOnePhysics
/// @brief problem evaluator class for elliptic, single-physics problems
/// @tparam PhysicsType defines physics and related quantity of interests for this physics 
template<typename PhysicsType>
class ProblemEvaluatorOnePhysics : public Plato::ProblemEvaluatorBase
{
private:
  /// @brief local criterion typename
  using Criterion = std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>;
  /// @brief local typename for map from criterion name to criterion evaluator
  using Criteria  = std::map<std::string, Criterion>;
  /// @brief local typename for physics-based element interface
  using ElementType     = typename PhysicsType::ElementType;
  /// @brief local typename for topological element
  using TopoElementType = typename ElementType::TopoElementType;
  /// @brief local typename for vector function evaluator
  using VectorFunctionType = Plato::Elliptic::VectorFunction<PhysicsType>;
  /// @brief contains mesh and model information
  Plato::SpatialModel & mSpatialModel;
  /// @brief output database
  Plato::DataMap & mDataMap;
  /// @brief residual evaluator
  std::shared_ptr<VectorFunctionType> mResidualEvaluator; 
  /// @brief map from criterion name to criterion evaluator
  Criteria mCriterionEvaluator;
  /// @brief number of newton steps/cycles
  Plato::OrdinalType mNumNewtonSteps;
  /// @brief residual and increment tolerances for newton solver
  Plato::Scalar mNewtonResTol, mNewtonIncTol;
    /// @brief apply dirichlet boundary condition weakly
  bool mWeakEBCs = false;
  /// @brief vector of state values
  Plato::ScalarMultiVector mStates; 
  /// @brief dirichlet degrees of freedom
  Plato::OrdinalVector mDirichletDofs; 
  /// @brief dirichlet degrees of freedom state values
  Plato::ScalarVector mDirichletStateVals; 
  /// @brief dirichlet degrees of freedom adjoint values
  Plato::ScalarVector mDirichletAdjointVals;
  /// @brief multipoint constraint interface
  std::shared_ptr<Plato::MultipointConstraints> mMPCs;
  /// @brief linear solver interface
  std::shared_ptr<Plato::AbstractSolver> mSolver;
  /// @brief partial differential equation type
  std::string mTypePDE; 
  /// @brief simulated physics
  std::string mPhysics; 

public:
  /// @brief class constructor
  /// @param [in] aParamList    input problem parameters
  /// @param [in] aSpatialModel contains mesh and model information
  /// @param [in] aDataMap      output database
  /// @param [in] aMachine      mpi wrapper
  ProblemEvaluatorOnePhysics(
    Teuchos::ParameterList & aParamList,
    Plato::SpatialModel    & aSpatialModel,
    Plato::DataMap         & aDataMap,
    Plato::Comm::Machine     aMachine
  );

  /// @brief class destructor
  ~ProblemEvaluatorOnePhysics(){}

  /// @fn getSolution
  /// @brief get state solution
  /// @return solutions database
  Plato::Solutions
  getSolution();

  /// @fn criterionIsLinear
  /// @brief return true if criterion is linear; otherwise, return false
  /// @param [in] aName criterion name
  /// @return boolean
  bool
  criterionIsLinear(
    const std::string & aName
  );

  /// @fn analyze
  /// @brief solve single-physics problem, solution is saved into the range and domain database
  /// @param [in,out] aDatabase range and domain database
  void
  analyze(
    Plato::Database & aDatabase
  );

  /// @fn residual
  /// @brief evaluate residual for a single-physics problem, residual is save into the database
  /// @param [in,out] aDatabase range and domain database
  void
  residual(
    Plato::Database & aDatabase
  );

  /// @fn criterionValue
  /// @brief evaluate criterion for a single-physics problem
  /// @param [in]     aName     criterion name
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar
  Plato::Scalar
  criterionValue(
    const std::string     & aName,
          Plato::Database & aDatabase
  );

  /// @fn criterionGradient
  /// @brief compute criterion gradient for a single-physics problem
  /// @param [in]     aName     criterion name
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar
  Plato::ScalarVector
  criterionGradient(
    const Plato::evaluation_t & aEvalType,
    const std::string         & aName,
          Plato::Database     & aDatabase
  );

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in,out] aDatabase range and domain database
  void 
  updateProblem(
    Plato::Database & aDatabase
  );

private:
  /// @fn updateDatabase
  /// @brief update range and domain database by appending state containers
  /// @param [in,out] aDatabase range and domain database
  void
  updateDatabase(
    Plato::Database & aDatabase
  );

  /// @fn getErrorMsg
  /// @brief get error message in the case a cirterion is not defined
  /// @param [in] aName criterion name
  /// @return string
  std::string
  getErrorMsg(
    const std::string & aName
  ) const;

  /// @fn criterionGradientControl
  /// @brief compute criterion gradient with respect to the controls
  /// @param [in]     aCriterion criterion name
  /// @param [in,out] aDatabase  range and domain database
  /// @return scalar vector
  Plato::ScalarVector 
  criterionGradientControl(
    const Criterion       & aCriterion,
          Plato::Database & aDatabase
  );

  /// @fn criterionGradientControl
  /// @brief compute criterion gradient with respect to the configuration
  /// @param [in]     aCriterion criterion name
  /// @param [in,out] aDatabase  range and domain database
  /// @return scalar vector
  Plato::ScalarVector 
  criterionGradientConfig(
    const Criterion       & aCriterion,
          Plato::Database & aDatabase
  );

  /// @fn initializeEvaluators
  /// @brief initialize criteria and residual evaluators
  /// @param [in] aParamList input problem parameters 
  void 
  initializeEvaluators(
    Teuchos::ParameterList& aParamList
  );

  /// @fn readEssentialBoundaryConditions
  /// @brief read essential boundary conditions information describing which degrees of freedom 
  /// are constrained and corresponding constraint values
  /// @param [in] aParamList input problem parameters
  void 
  readEssentialBoundaryConditions(
    Teuchos::ParameterList & aParamList
  );

  /// @fn setEssentialBoundaryConditions
  /// @brief set essential boundary condition information 
  /// @param [in] aDofs   constrained degrees of freedom (dofs)
  /// @param [in] aValues values associated with constrained dofs
  void 
  setEssentialBoundaryConditions(
    const Plato::OrdinalVector & aDofs, 
    const Plato::ScalarVector  & aValues
  );

  /// @fn initializeMultiPointConstraints
  /// @brief initialize multi-point constraints interface 
  /// @param [in] aParamList input problem parameters
  void 
  initializeMultiPointConstraints(
    Teuchos::ParameterList & aParamList
  );

  /// @brief initialize linear solver interface
  /// @param aMesh      mesh database
  /// @param aParamList input problem parameters
  /// @param aMachine   mpi wrapper
  void 
  initializeSolver(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aParamList,
    Comm::Machine          & aMachine
  );

  /// @fn enforceStrongEssentialBoundaryConditions
  /// @brief enforce strong essential boundary conditions to linear system of equation
  /// @param [in,out] aMatrix matrix
  /// @param [in,out] aVector right-hand-side vector
  /// @param [in] aMultiplier scalar multiplier
  void
  enforceStrongEssentialBoundaryConditions(
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVector                & aVector,
    const Plato::Scalar                      & aMultiplier
  );

  /// @fn enforceStrongEssentialAdjointBoundaryConditions
  /// @brief enforce strong essential boundary conditions to adjoint linear system of equation
  /// @param [in,out] aMatrix matrix
  /// @param [in,out] aVector right-hand-side vector
  void 
  enforceStrongEssentialAdjointBoundaryConditions(
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVector                & aVector
  );

  /// @fn enforceWeakEssentialAdjointBoundaryConditions
  /// @brief allocate dirichelt container with constraint values to allow weak enforcement of the 
  /// dirichlet boundary conditions for the adjoint problem
  /// @param [in,out] aDatabase range and domain database
  void 
  enforceWeakEssentialAdjointBoundaryConditions(
    Plato::Database & aDatabase
  );

};

} // namespace Elliptic

} // namespace Plato
