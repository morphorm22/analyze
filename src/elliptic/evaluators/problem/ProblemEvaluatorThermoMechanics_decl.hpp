/*
 * ProblemEvaluatorThermoMechanics_decl.hpp
 *
 *  Created on: June 27, 2023
 */

#pragma once

#include "Solutions.hpp"
#include "SpatialModel.hpp"

#include "base/Database.hpp"
#include "base/ProblemEvaluatorBase.hpp"
#include "base/SupportedParamOptions.hpp"

#include "solver/PlatoAbstractSolver.hpp"

#include "elliptic/base/VectorFunctionBase.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorBase.hpp"
#include "elliptic/evaluators/problem/SupportedEllipticProblemOptions.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class ProblemEvaluatorThermoMechanics
/// @brief coordinates the thermomechanical forward and adjoint physics evaluations plus 
/// the criteria value and gradient evaluations
/// @tparam PhysicsType defines physics and related physical quantity of interests for this problem
template<typename PhysicsType>
class ProblemEvaluatorThermoMechanics : public Plato::ProblemEvaluatorBase
{
private:
  /// @brief local criterion typename
  using Criterion = std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>;
  /// @brief local typename for map from criterion name to criterion evaluator
  using Criteria = std::unordered_map<std::string, Criterion>;
  /// @brief local typename for physics-based element interface
  using ElementType = typename PhysicsType::ElementType;
  /// @brief local typename for topological element
  using TopoElementType = typename ElementType::TopoElementType;
  /// @brief local typename for thermal physics type
  using ThermalPhysicsType = typename PhysicsType::OtherPhysics;
  /// @brief local typename for vector function evaluator
  using Residual = std::shared_ptr<Plato::Elliptic::VectorFunctionBase>;
  /// @brief local typename for map from residual name to residual evaluator
  using Residuals = std::unordered_map<Plato::Elliptic::residual_t,Residual>;
  /// @brief local typename for linear system solver interface
  using Solver = std::shared_ptr<Plato::AbstractSolver>;
  /// @brief local typename for map from residual type name to linear system solver interface
  using Solvers = std::unordered_map<Plato::Elliptic::residual_t,Solver>;
  
  /// @brief contains mesh and model information
  Plato::SpatialModel & mSpatialModel;
  /// @brief output database
  Plato::DataMap & mDataMap;
  /// @brief map from residual type to mechanical residual evaluator
  Residuals mResidualEvaluators;
  /// @brief map from criterion name to thermal criterion evaluator
  Criteria mCriterionEvaluators;
  /// @brief map from residual type to linear system solver interface
  Solvers mLinearSolvers;

  /// @brief number of newton steps/cycles
  Plato::OrdinalType mNumNewtonSteps;
  /// @brief residual and increment tolerances for newton solver
  Plato::Scalar mNewtonResTol, mNewtonIncTol;

  /// @brief apply dirichlet boundary condition weakly
  bool mWeakEBCs = false;

  /// @brief vector state values
  Plato::ScalarMultiVector mTemperatures; 
  /// @brief node state values
  Plato::ScalarMultiVector mDisplacements; 

  /// @brief map from residual type enum to dirichlet degrees of freedom
  std::unordered_map<Plato::Elliptic::residual_t,Plato::OrdinalVector> mDirichletDofs;
  /// @brief map from residual type enum to dirichlet degrees of freedom values
  std::unordered_map<Plato::Elliptic::residual_t,Plato::ScalarVector> mDirichletVals;

  /// @brief partial differential equation type
  std::string mTypePDE; 
  /// @brief simulated physics
  std::string mPhysics; 

  /// @brief requested thermal residual type
  Plato::Elliptic::residual_t mThermalResidualType;
  /// @brief requested mechanical residual type
  Plato::Elliptic::residual_t mMechanicalResidualType;

public:
  /// @brief class constructor
  /// @param [in] aParamList    input problem parametes
  /// @param [in] aSpatialModel contains mesh and model information
  /// @param [in] aDataMap      output database
  /// @param [in] aMachine      mpi wrapper 
  ProblemEvaluatorThermoMechanics(
    Teuchos::ParameterList & aParamList,
    Plato::SpatialModel    & aSpatialModel,
    Plato::DataMap         & aDataMap,
    Plato::Comm::Machine     aMachine
  );

  /// @brief class destructor
  ~ProblemEvaluatorThermoMechanics(){}

  /// @fn getSolution
  /// @brief get state solution
  /// @return solutions database
  Plato::Solutions
  getSolution();

  /// @fn postProcess
  /// @brief post process solution database before output
  /// @param [in] aSolutions solution database
  void
  postProcess(
    Plato::Solutions & aSolutions
  );

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in,out] aDatabase range and domain database
  void 
  updateProblem(
    Plato::Database & aDatabase
  );

  /// @fn analyze
  /// @brief analyze physics, solution is saved into the database
  /// @param [in,out] aDatabase range and domain database
  void
  analyze(
    Plato::Database & aDatabase
  );

  /// @fn residual
  /// @brief evaluate thermomechanical residual, residual is save into the database
  /// @param [in,out] aDatabase range and domain database
  void
  residual(
    Plato::Database & aDatabase
  );

  /// @fn criterionValue
  /// @brief evaluate criterion 
  /// @param [in]     aName     criterion name
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar
  Plato::Scalar
  criterionValue(
    const std::string     & aName,
          Plato::Database & aDatabase
  );

  /// @fn criterionGradient
  /// @brief compute criterion gradient
  /// @param [in]     aEvalType evaluation type, compute gradient with respect to a quantity of interests
  /// @param [in]     aName     criterion name
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar vector
  Plato::ScalarVector
  criterionGradient(
    const Plato::evaluation_t & aEvalType,
    const std::string         & aName,
          Plato::Database     & aDatabase
  );

  /// @fn criterionIsLinear
  /// @brief return true if criterion is linear; otherwise, return false
  /// @param [in] aName criterion name
  /// @return boolean
  bool  
  criterionIsLinear(
    const std::string & aName
  );

private:
  /// @fn analyzeThermalPhysics
  /// @brief solve thermal linear system of equations
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar vector
  void
  analyzeThermalPhysics(
    Plato::Database & aDatabase
  );

  /// @fn analyzeMechanicalPhysics
  /// @brief solve mechanical linear system of equations
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar vector
  void
  analyzeMechanicalPhysics(
    Plato::Database & aDatabase
  );

  /// @fn analyzePhysics
  /// @brief solve forward problem for the physics of interests and return solution
  /// @param [in]     aResidual residual evaluator
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar vector
  Plato::ScalarVector 
  analyzePhysics(
    const Residual        & aResidual,
    const Plato::Database & aDatabase
  );

  /// @fn enforceStrongEssentialBoundaryConditions
  /// @brief enforce strong dirichlet boundary conditions on forward problem \f$\mathbf{A}=\mathbf{f}\f$,
  /// where \f$\mathbf{A}\f$ is the left hand side matrix and \f$\mathbf{f}\f$ is the right hand side vector
  /// @param [in]     aMultiplier scalar 
  /// @param [in]     aResidual   residual evaluator
  /// @param [in,out] aMatrix     left hand side matrix
  /// @param [in,out] aVector     right hand side vector
  void 
  enforceStrongEssentialBoundaryConditions(
    const Plato::Scalar                      & aMultiplier,
    const Residual                           & aResidual,
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVector                & aVector
  );

  /// @fn adjointThermalPhysics
  /// @brief compute adjoints for thermal physics
  /// @param [in]     aCriterion    criterion evaluator
  /// @param [in]     aMechAdjoints mechanical adjoints
  /// @param [in,out] aDatabase     range and domain database
  /// @return scalar vector
  Plato::ScalarVector 
  adjointThermalPhysics(
    const Criterion           & aCriterion,
    const Plato::ScalarVector & aMechAdjoints,
          Plato::Database     & aDatabase
  );

  /// @fn adjointMechanicalPhysics
  /// @brief compute adjoints for mechanical physics
  /// @param [in]     aCriterion criterion evaluator
  /// @param [in,out] aDatabase  range and domain database
  /// @return scalar vector
  Plato::ScalarVector
  adjointMechanicalPhysics(
    const Criterion       & aCriterion,
          Plato::Database & aDatabase
  );

  /// @fn criterionGradientControl
  /// @brief compute criterion with respect to the controls
  /// @param [in]     aCriterion criterion evaluator
  /// @param [in,out] aDatabase  range and domain database
  /// @return scalar vector
  Plato::ScalarVector 
  criterionGradientControl(
    const Criterion       & aCriterion,
          Plato::Database & aDatabase
  );

  /// @fn criterionGradientConfig
  /// @brief compute criterion with respect to the configuration
  /// @param [in]     aCriterion criterion evaluator
  /// @param [in,out] aDatabase  range and domain database
  /// @return scalar vector
  Plato::ScalarVector 
  criterionGradientConfig(
    const Criterion       & aCriterion,
          Plato::Database & aDatabase
  );

  /// @fn enforceStrongEssentialAdjointBoundaryConditions
  /// @brief enforce strong essential boundary conditions on adjoint problem \f$\mathbf{A}^{T}=\mathbf{f}\f$,
  /// where \f$\mathbf{A}\f$ is the left hand side matrix and \f$\mathbf{f}\f$ is the right hand side vector
  /// @param [in]     aResidual residual evaluator
  /// @param [in,out] aMatrix   left hand side matrix 
  /// @param [in,out] aVector   right hand side vecrtor
  void 
  enforceStrongEssentialAdjointBoundaryConditions(
    const Residual                           & aResidual,
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVector                & aVector
  );

  /// @fn enforceWeakEssentialAdjointBoundaryConditions
  /// @brief enforce weak essential boundary conditions on adjoint problem \f$\mathbf{A}^{T}=\mathbf{f}\f$,
  /// where \f$\mathbf{A}\f$ is the left hand side matrix and \f$\mathbf{f}\f$ is the right hand side vector
  /// @param [in]     aResidual residual evaluator
  /// @param [in,out] aDatabase range and domain database
  void 
  enforceWeakEssentialAdjointBoundaryConditions(
    const Residual        & aResidual,
          Plato::Database & aDatabase
  );

  /// @fn setSolution
  /// @brief set solutions database
  /// @param [in]     aResidual  residual evaluator
  /// @param [in,out] aSolutions solutions database
  void 
  setSolution(
    const Residual         & aResidual,
          Plato::Solutions & aSolutions
  );

  /// @fn getCriterionErrorMsg
  /// @brief get error message if requested criterion parameter list is undefined
  /// @param [in] aName criterion paramater list name
  /// @return error message string
  std::string
  getCriterionErrorMsg(
    const std::string & aName
  ) const;

  /// @fn updateDatabase
  /// @brief update range and domain database
  /// @param [in,out] aDatabase range and domain database
  void 
  updateDatabase(
    Plato::Database & aDatabase
  );

  /// @fn initializeCriterionEvaluators
  /// @brief initialize criterion evaluators
  /// @param [in] aParamList input problem parameters
  void 
  initializeCriterionEvaluators(
    Teuchos::ParameterList & aParamList
  );

  /// @fn initializeThermalResidualEvaluators
  /// @brief initialize thermal residual evaluator
  /// @param [in] aParamList input problem parameters
  void 
  initializeThermalResidualEvaluators(
    Teuchos::ParameterList & aParamList
  );

  /// @fn initializeMechanicalResidualEvaluators
  /// @brief initialize mechanical residual evaluator
  /// @param [in] aParamList input problem parameters
  void 
  initializeMechanicalResidualEvaluators(
    Teuchos::ParameterList & aParamList
  );

  /// @fn initializeSolvers
  /// @brief create solver interface
  /// @param [in] aMesh      mesh database
  /// @param [in] aParamList input problem parameters
  /// @param [in] aMachine   mpi wrapper
  void 
  initializeSolvers(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aParamList,
    Comm::Machine          & aMachine
  );

  /// @fn readEssentialBoundaryConditions
  /// @brief read input thermal essential boundary conditions
  /// @param [in] aParamList 
  void 
  readEssentialBoundaryConditions(
    Teuchos::ParameterList & aParamList
  );

  /// @fn readMechanicalEssentialBoundaryConditions
  /// @brief read input thermal essential boundary conditions
  /// @param [in] aParamList 
  void 
  readMechanicalEssentialBoundaryConditions(
    Teuchos::ParameterList & aParamList
  );

  /// @fn readThermalEssentialBoundaryConditions
  /// @brief read input thermal essential boundary conditions
  /// @param [in] aParamList 
  void 
  readThermalEssentialBoundaryConditions(
    Teuchos::ParameterList & aParamList
  );

};

} // namespace Elliptic

} // namespace Plato