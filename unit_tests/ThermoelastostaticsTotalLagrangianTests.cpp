/*
 * ThermoelastostaticTotalLagrangianTests.cpp
 *
 *  Created on: June 14, 2023
 */

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// analyze includes
#include "Tri3.hpp"
#include "PlatoMathTypes.hpp"
#include "ApplyConstraints.hpp"
#include "InterpolateFromNodal.hpp"

#include "element/ThermoElasticElement.hpp"

#include "Tet10.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/base/WorksetBuilder.hpp"
#include "elliptic/mechanical/nonlinear/NominalStressTensor.hpp"
#include "elliptic/mechanical/nonlinear/KineticPullBackOperation.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermalDeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermoElasticDeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ResidualThermoElastoStaticTotalLagrangian.hpp"

#include "EssentialBCs.hpp"
#include "PlatoUtilities.hpp"
#include "elliptic/thermal/Thermal.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "base/ProblemEvaluatorBase.hpp"
#include "elliptic/base/VectorFunction.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorBase.hpp"
#include "elliptic/evaluators/criterion/FactoryCriterionEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

namespace thermomechanical
{

/// @enum residual
/// @brief supported residual enums for thermomechanical physics
enum struct residual
{
  LINEAR_THERMO_MECHANICS=0,
  NONLINEAR_THERMO_MECHANICS=1
};

/// @struct ResidualEnum
/// @brief Interface between input state response type input and supported thermomechanical residual 
struct ResidualEnum
{
private:
  /// @brief map from state response type to supported thermomechanical residual enum
  std::unordered_map<std::string,Plato::Elliptic::thermomechanical::residual> s2e = 
  {
    {"linear"   ,Plato::Elliptic::thermomechanical::residual::LINEAR_THERMO_MECHANICS},
    {"nonlinear",Plato::Elliptic::thermomechanical::residual::NONLINEAR_THERMO_MECHANICS}
  };

public:
  /// @brief return supported  elliptic thermomechanical residual enum
  /// @param [in] aResponse state response, linear or nonlinear
  /// @return residual enum
  Plato::Elliptic::thermomechanical::residual 
  get(
    const std::string & aResponse
  ) 
  const
  {
    auto tLowerResponse = Plato::tolower(aResponse);
    auto tItrResponse = s2e.find(tLowerResponse);
    if( tItrResponse == s2e.end() ){
      auto tMsg = this->getErrorMsg(tLowerResponse);
      ANALYZE_THROWERR(tMsg)
    }
    return tItrResponse->second;
  }

private:
  /// @fn getErrorMsg
  /// @brief Return error message if response is not supported
  /// @param [in] aResponse string - response type, linear or nonlinear
  /// @return error message string
  std::string
  getErrorMsg(
    const std::string & aResponse
  )
  const
  {
    auto tMsg = std::string("Did not find response '") + aResponse 
      + "'. Supported response options are: ";
    for(const auto& tPair : s2e){
      tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
  }
};

} // namespace thermomechanical

namespace NonlinearThermoMechanics
{

struct FunctionFactory
{
  template<typename EvaluationType>
  std::shared_ptr<Plato::ResidualBase>
  createVectorFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList,
          std::string              aTypePDE
  )
  {  
    if( !aParamList.sublist(aTypePDE).isSublist("Mechanical Residual") ){ 
      ANALYZE_THROWERR("ERROR: 'Mechanical Residual' parameter list not found!"); 
    }
    else{
      auto tMechResidualParamList = aParamList.sublist(aTypePDE).sublist("Mechanical Residual");
      auto tResponse = tMechResidualParamList.get<std::string>("Response","linear");
      Plato::Elliptic::thermomechanical::ResidualEnum tSupportedResidual;
      auto tResidual = tSupportedResidual.get(tResponse);
      switch (tResidual)
      {
      case Plato::Elliptic::thermomechanical::residual::NONLINEAR_THERMO_MECHANICS:
        return 
          (std::make_shared<Plato::Elliptic::ResidualThermoElastoStaticTotalLagrangian<EvaluationType>>(
            aSpatialDomain, aDataMap, aParamList
          ));
        break;
      case Plato::Elliptic::thermomechanical::residual::LINEAR_THERMO_MECHANICS:
      default:
        ANALYZE_THROWERR("ERROR: Requested 'Mechanical Residual' is not supported!"); 
        break;
      }
    }
  }

  template<typename EvaluationType>
  std::shared_ptr<Plato::CriterionBase>
  createScalarFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap, 
          Teuchos::ParameterList & aProblemParams, 
          std::string              aFuncType,
          std::string              aFuncName
  )
  {  
    ANALYZE_THROWERR("ERROR: Requested 'Scalar Function' is not supported");
  }

};

} // namespace NonlinearThermoMechanics

namespace Nonlinear
{

/// @brief concrete class use to define elliptic thermomechanics physics
/// @tparam TopoElementType topological element typename 
template<typename TopoElementType>
class ThermoMechanics : public Plato::ThermoElasticElement<TopoElementType>
{
public:
  /// @brief residual and criteria factory for elliptic thermomechanics physics
  typedef Plato::Elliptic::NonlinearThermoMechanics::FunctionFactory FunctionFactory;
  /// @brief topological element type with additional physics related information 
  using ElementType = ThermoElasticElement<TopoElementType>;
  /// @brief typename for linear thermal physics
  using ThermalPhysics = Plato::Elliptic::Linear::Thermal<TopoElementType>;
};

} // namespace Nonlinear

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
  using ThermalPhysicsType = typename PhysicsType::ThermalPhysics;
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

  Plato::Elliptic::residual_t mThermalResidualType;
  Plato::Elliptic::residual_t mMechanicalResidualType;

public:
  ProblemEvaluatorThermoMechanics(
    Teuchos::ParameterList & aParamList,
    Plato::SpatialModel    & aSpatialModel,
    Plato::DataMap         & aDataMap,
    Plato::Comm::Machine     aMachine
  ) : 
    mSpatialModel  (aSpatialModel),
    mDataMap       (aDataMap),
    mNumNewtonSteps(Plato::ParseTools::getSubParam<int>(aParamList,"Newton Iteration","Maximum Iterations",1.)),
    mNewtonIncTol  (Plato::ParseTools::getSubParam<double>(aParamList,"Newton Iteration","Increment Tolerance",0.)),
    mNewtonResTol  (Plato::ParseTools::getSubParam<double>(aParamList,"Newton Iteration","Residual Tolerance",0.)),
    mTypePDE       (aParamList.get<std::string>("PDE Constraint")),
    mPhysics       (aParamList.get<std::string>("Physics"))
  {
    this->initializeThermalResidualEvaluators(aParamList);
    this->initializeMechanicalResidualEvaluators(aParamList);
    this->initializeCriterionEvaluators(aParamList);
    this->readEssentialBoundaryConditions(aParamList);
    this->initializeSolvers(mSpatialModel.Mesh,aParamList,aMachine);
  }

  /// @fn getSolution
  /// @brief get state solution
  /// @return solutions database
  Plato::Solutions
  getSolution()
  {
    Plato::Solutions tSolution(mPhysics, mTypePDE);
    for( auto& tPair : mResidualEvaluators ){
      this->setSolution(tPair.second,tSolution);
    }
    return tSolution;
  }

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in,out] aDatabase range and domain database
  void 
  updateProblem(
    Plato::Database & aDatabase
  )
  {
    this->updateDatabase(aDatabase);
    const Plato::Scalar tCycle = aDatabase.scalar("cycle");
    for( auto& tCriterion : mCriterionEvaluators ) {
      tCriterion.second->updateProblem(aDatabase, tCycle);
    }
  }

  /// @fn analyze
  /// @brief analyze physics of interests, solution is saved into the database
  /// @param [in,out] aDatabase range and domain database
  void
  analyze(
    Plato::Database & aDatabase
  )
  {
    // analyze thermal equations
    this->analyzeThermalPhysics(aDatabase);
    // analyze mechanical physics
    this->analyzeMechanicalPhysics(aDatabase);
  }

  /// @fn residual
  /// @brief evaluate thermomechanical residual, residual is save into the database
  /// @param [in,out] aDatabase range and domain database
  void
  residual(
    Plato::Database & aDatabase
  )
  {
    this->updateDatabase(aDatabase);
    const Plato::Scalar tCycle = aDatabase.scalar("cycle");
    // residuals are not saved in the database
    for(auto& tPair : mResidualEvaluators){
      auto tResidual = tPair.second->value(aDatabase,tCycle);
    }
  }

  /// @fn criterionValue
  /// @brief evaluate criterion 
  /// @param [in]     aName     criterion name
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar
  Plato::Scalar
  criterionValue(
    const std::string     & aName,
          Plato::Database & aDatabase
  )
  {
    if( mCriterionEvaluators.find(aName) == mCriterionEvaluators.end() )
    {
      auto tErrMsg = this->getCriterionErrorMsg(aName);
      ANALYZE_THROWERR(tErrMsg)
    }
    this->updateDatabase(aDatabase);
    const Plato::Scalar tCycle = aDatabase.scalar("cycle");
    auto tValue = mCriterionEvaluators[aName]->value(aDatabase,tCycle);
    return tValue;
  }

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
  )
  {
    auto tCriterionItr = mCriterionEvaluators.find(aName);
    if( tCriterionItr == mCriterionEvaluators.end() )
    {
      auto tErrMsg = this->getErrorMsg(aName);
      ANALYZE_THROWERR(tErrMsg)
    }
    this->updateDatabase(aDatabase);
    switch (aEvalType)
    {
      case Plato::evaluation_t::GRAD_Z: 
        return ( this->criterionGradientControl(tCriterionItr.operator*(),aDatabase) );
        break;
      case Plato::evaluation_t::GRAD_X:
        return ( this->criterionGradientConfig(tCriterionItr.operator*(),aDatabase) );
        break;
      default:
        return ( Plato::ScalarVector("empty",0) );
        break;
    }
  }

  /// @fn criterionIsLinear
  /// @brief return true if criterion is linear; otherwise, return false
  /// @param [in] aName criterion name
  /// @return boolean
  bool  
  criterionIsLinear(
    const std::string & aName
  )
  {
    if( mCriterionEvaluators.find(aName) == mCriterionEvaluators.end() )
    {
      auto tErrMsg = this->getCriterionErrorMsg(aName);
      ANALYZE_THROWERR(tErrMsg)
    }
    return ( mCriterionEvaluators.at(aName)->isLinear() );
  }

private:
  void
  analyzeThermalPhysics(
    Plato::Database & aDatabase
  )
  {
    auto tResidualItr = mResidualEvaluators.find(mThermalResidualType);
    if(tResidualItr == mResidualEvaluators.end() ){
      ANALYZE_THROWERR("ERROR: Did not find requested thermal residual evaluator")
    }
    auto tTemperatures = this->analyzePhysics(tResidualItr.operator*(),aDatabase);
    aDatabase.vector("node states",tTemperatures);
  }

  Plato::ScalarVector 
  analyzeMechanicalPhysics(
    Plato::Database & aDatabase
  )
  {
    auto tResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
    if(tResidualItr == mResidualEvaluators.end() ){
      ANALYZE_THROWERR("ERROR: Did not find requested mechanical residual evaluator")
    }
    auto tTemperatures = this->analyzePhysics(tResidualItr.operator*(),aDatabase);
    aDatabase.vector("node states",tTemperatures);
  }

  Plato::ScalarVector 
  analyzePhysics(
    const Residual        & aResidual,
    const Plato::Database & aDatabase
  )
  {
    // initialize displacements values
    Plato::Database tDatabase(aDatabase);
    auto tNumStates = mSpatialModel.Mesh->NumNodes() * aResidual->numStateDofsPerNode();
    Plato::ScalarVector tStates("states",tNumStates);
    Plato::blas1::fill(0.0, tStates);
    tDatabase.vector("states",tStates);
    // inner loop for non-linear models
    const Plato::Scalar tCycle = aDatabase.scalar("cycle");
    for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
    {
      Plato::ScalarVector tResidual = aResidual->value(aDatabase,tCycle);
      Plato::blas1::scale(-1.0, tResidual);
      if (mNumNewtonSteps > 1) {
        auto tResidualNorm = Plato::blas1::norm(tResidual);
        std::cout << " Residual norm: " << tResidualNorm << std::endl;
        if (tResidualNorm < mNewtonResTol) {
          std::cout << " Residual norm tolerance satisfied." << std::endl;
          break;
        }
      }
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianState = 
        aResidual->jacobianState(aDatabase,tCycle,/*transpose=*/false);
      // solve linear system of equations
      Plato::Scalar tScale = (tNewtonIndex == 0) ? 1.0 : 0.0;
      if( !mWeakEBCs )
      { this->enforceStrongEssentialBoundaryConditions(aResidual,tJacobianState,tResidual,tScale); }
      Plato::ScalarVector tDelta("increment", tStates.extent(0));
      Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDelta);
      mLinearSolvers[aResidual->type()]->solve(*tJacobianState, tDelta, tResidual);
      Plato::blas1::axpy(1.0, tDelta, tStates);
      if (mNumNewtonSteps > 1) {
        auto tIncrementNorm = Plato::blas1::norm(tDelta);
        std::cout << " Delta norm: " << tIncrementNorm << std::endl;
        if (tIncrementNorm < mNewtonIncTol) {
          std::cout << " Solution increment norm tolerance satisfied." << std::endl;
          break;
        }
      }
    }
    return tStates;
  }

  void 
  enforceStrongEssentialBoundaryConditions(
    const Residual                           & aResidual,
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVector                & aVector,
    const Plato::Scalar                      & aMultiplier
  )
  {
    if(aMatrix->isBlockMatrix())
    {
      Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(
        aMatrix, aVector, mDirichletDofs[aResidual->type()], mDirichletVals[aResidual->type()], aMultiplier
      );
    }
    else
    {
      Plato::applyConstraints<ElementType::mNumDofsPerNode>(
        aMatrix, aVector, mDirichletDofs[aResidual->type()], mDirichletVals[aResidual->type()], aMultiplier
      );
    }
  }

  Plato::ScalarVector 
  adjointThermalPhysics(
    const Criterion           & aCriterion,
    const Plato::Database     & aDatabase,
    const Plato::ScalarVector & aMechAdjoints
  )
  {
    // compute gradient with respect to node states
    const Plato::Scalar tCycle = aDatabase.scalar("cycle");
    auto tRightHandSideVector = aCriterion->gradientNodeState(aDatabase,tCycle);
    // compute jacobian of mechanical residual with respect to node states
    auto tMechanicalResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
    Teuchos::RCP<Plato::CrsMatrixType> tMechJacobianNodeState = 
      tMechanicalResidualItr->second->jacobianNodeState(aDatabase,tCycle,/*transpose=*/true);
    // add mechanical residual contribution to right hand side vector 
    Plato::MatrixTimesVectorPlusVector(tMechJacobianNodeState,aMechAdjoints,tRightHandSideVector);
    // compute additive inverse of the right hand side vector 
    Plato::blas1::scale(-1.0,tRightHandSideVector);
    // compute jacobian of node state residual with respect to node states
    auto tThermalResidualItr = mResidualEvaluators.find(mThermalResidualType);
    Teuchos::RCP<Plato::CrsMatrixType> tTempJacobianState = 
      tThermalResidualItr->second->jacobianState(aDatabase,tCycle,/*transpose=*/true);
    // enforce dirichlet boundary conditons 
    if( mWeakEBCs ){ 
      this->enforceWeakEssentialAdjointBoundaryConditions(tThermalResidualItr.operator*(),aDatabase); 
    }
    else{ 
      this->enforceStrongEssentialAdjointBoundaryConditions(
        tThermalResidualItr.operator*(),tTempJacobianState,tRightHandSideVector
      ); 
    }
    // solve adjoint system of equations
    auto tNumTempStates = mSpatialModel.Mesh->NumNodes() * tThermalResidualItr->second->numStateDofsPerNode();
    Plato::ScalarVector tTempAdjoints("thermal adjoints",tNumTempStates);
    mLinearSolvers[tThermalResidualItr->second->type()]->solve(
      *tTempJacobianState,tTempAdjoints,tRightHandSideVector,/*isAdjointSolve=*/true
    );
    return tTempAdjoints;
  }

  Plato::ScalarVector
  adjointMechanicalPhysics(
    const Criterion       & aCriterion,
    const Plato::Database & aDatabase
  )
  {
    auto tResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
    // compute gradient with respect to state variables
    const Plato::Scalar tCycle = aDatabase.scalar("cycle");
    auto tGradientState = aCriterion->gradientState(aDatabase,tCycle);
    Plato::blas1::scale(-1.0, tGradientState);
    // compute jacobian with respect to state variables
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianState = 
      tResidualItr->second->jacobianState(aDatabase,tCycle,/*transpose=*/true);
    // enforce dirichlet boundary conditons 
    if( mWeakEBCs )
    { this->enforceWeakEssentialAdjointBoundaryConditions(tResidualItr.operator*(),aDatabase); }
    else
    { this->enforceStrongEssentialAdjointBoundaryConditions(tResidualItr.operator*(),tJacobianState,tGradientState); }
    // solve adjoint system of equations
    auto tNumStates = mSpatialModel.Mesh->NumNodes() * tResidualItr->second->numStateDofsPerNode();
    Plato::ScalarVector tAdjoints("Adjoint Variables",tNumStates);
    mLinearSolvers[tResidualItr->second->type()]->solve(
      *tJacobianState,tAdjoints,tGradientState,/*isAdjointSolve=*/true
    );
    return tAdjoints;
  }

  Plato::ScalarVector 
  criterionGradientControl(
    const Criterion       & aCriterion,
          Plato::Database & aDatabase
  )
  {
    if(aCriterion == nullptr){ 
      ANALYZE_THROWERR("ERROR: Requested criterion is a null pointer"); 
    }
    // compute criterion contribution to the gradient
    const Plato::Scalar tCycle = aDatabase.scalar("cycle");
    auto tGradientControl = aCriterion->gradientControl(aDatabase,tCycle);
    // add residual contribution to the gradient
    if( aCriterion->isLinear() == false )
    {
      // compute mechanical adjoints
      Plato::ScalarVector tMechAdjoints = this->adjointMechanicalPhysics(aCriterion,aDatabase);
      // compute jacobian of mechanical residual with respect to the controls
      auto tMechResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
      Teuchos::RCP<Plato::CrsMatrixType> tMechJacobianControl = 
        tMechResidualItr->second->jacobianControl(aDatabase,tCycle,/*transpose=*/true);
      // add mechanical residual contribution to gradient with respect to the controls
      Plato::MatrixTimesVectorPlusVector(tMechJacobianControl,tMechAdjoints,tGradientControl);
      // compute thermal adjoints
      Plato::ScalarVector tTempAdjoints = this->adjointThermalPhysics(aCriterion,aDatabase,tMechAdjoints);
      // compute jacobian of thermal residual with respect to the controls
      auto tThermalResidualItr = mResidualEvaluators.find(mThermalResidualType);
      Teuchos::RCP<Plato::CrsMatrixType> tTempJacobianControl = 
        tThermalResidualItr->second->jacobianControl(aDatabase,tCycle,/*transpose=*/true);
      // add thermal residual contribution to gradient with respect to the controls
      Plato::MatrixTimesVectorPlusVector(tTempJacobianControl,tTempAdjoints,tGradientControl);
    }
    return tGradientControl; 
  }

  Plato::ScalarVector 
  criterionGradientConfig(
    const Criterion       & aCriterion,
          Plato::Database & aDatabase
  )
  {
    if(aCriterion == nullptr){ 
      ANALYZE_THROWERR("ERROR: Requested criterion is a null pointer"); 
    }
    // compute criterion contribution to the gradient
    const Plato::Scalar tCycle = aDatabase.scalar("cycle");
    auto tGradientConfig = aCriterion->gradientConfig(aDatabase,tCycle);
    // add residual contribution to the gradient
    if( aCriterion->isLinear() == false )
    {
      // compute mechanical adjoints
      Plato::ScalarVector tMechAdjoints = this->adjointMechanicalPhysics(aCriterion,aDatabase);
      // compute jacobian of mechanical residual with respect to the controls
      auto tMechResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
      Teuchos::RCP<Plato::CrsMatrixType> tMechJacobianConfig = 
        tMechResidualItr->second->jacobianConfig(aDatabase,tCycle,/*transpose=*/true);
      // add mechanical residual contribution to gradient with respect to the controls
      Plato::MatrixTimesVectorPlusVector(tMechJacobianConfig,tMechAdjoints,tGradientConfig);
      // compute thermal adjoints
      Plato::ScalarVector tTempAdjoints = this->adjointThermalPhysics(aCriterion,aDatabase,tMechAdjoints);
      // compute jacobian of thermal residual with respect to the controls
      auto tThermalResidualItr = mResidualEvaluators.find(mThermalResidualType);
      Teuchos::RCP<Plato::CrsMatrixType> tTempJacobianConfig = 
        tThermalResidualItr->second->jacobianConfig(aDatabase,tCycle,/*transpose=*/true);
      // add thermal residual contribution to gradient with respect to the controls
      Plato::MatrixTimesVectorPlusVector(tTempJacobianConfig,tTempAdjoints,tGradientConfig);
    }
    return tGradientConfig; 
  }

void 
enforceStrongEssentialAdjointBoundaryConditions(
  const Residual                           & aResidual,
  const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
  const Plato::ScalarVector                & aVector
)
{
  // Essential Boundary Conditions (EBCs)
  auto tDirichletVals = mDirichletVals[aResidual->type()];
  Plato::ScalarVector tDirichletAdjointValues("Adjoint EBCs", tDirichletVals.size());
  Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tDirichletAdjointValues);
  if(aMatrix->isBlockMatrix())
  {
    Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(
      aMatrix, aVector, mDirichletDofs[aResidual->type()], tDirichletAdjointValues
    );
  }
  else
  {
    Plato::applyConstraints<ElementType::mNumDofsPerNode>(
      aMatrix, aVector, mDirichletDofs[aResidual->type()], tDirichletAdjointValues
    );
  }
}

  void 
  enforceWeakEssentialAdjointBoundaryConditions(
    const Residual        & aResidual,
          Plato::Database & aDatabase
  )
  {
    // Essential Boundary Conditions (EBCs)
    auto tNumStates = mSpatialModel.Mesh->NumNodes() * aResidual->numStateDofsPerNode();
    Plato::ScalarVector tAdjointDirichlet("Adjoint EBCs", tNumStates);
    Kokkos::deep_copy(tAdjointDirichlet, 0.0);
    aDatabase.vector("dirichlet",tAdjointDirichlet);
  }

  bool 
  isThermalPhysics(
    const Residual & aResidual
  )
  {
    if(aResidual->type() == Plato::Elliptic::residual_t::LINEAR_THERMAL){
      return true;
    }
    else{
      return false;
    }
  }

  bool 
  isMechanicalPhysics(
    const Residual & aResidual
  )
  {
    if( (aResidual->type() == Plato::Elliptic::residual_t::LINEAR_THERMO_MECHANICAL) || 
        (aResidual->type() == Plato::Elliptic::residual_t::NONLINEAR_THERMO_MECHANICAL) 
    )
    {
      return true;
    }
    else{
      return false;
    }
  }

  void 
  setSolution(
    const Residual         & aResidual,
          Plato::Solutions & aSolutions
  )
  {
    switch (aResidual->type())
    {
      case Plato::Elliptic::residual_t::LINEAR_THERMO_MECHANICAL:
      case Plato::Elliptic::residual_t::NONLINEAR_THERMO_MECHANICAL:
        aSolutions.set("Displacements",mDisplacements,aResidual->getDofNames());
        break;
      case Plato::Elliptic::residual_t::LINEAR_THERMAL:
        aSolutions.set("Temperatures" ,mTemperatures ,aResidual->getDofNames());
        break;
    }
  }

  std::string
  getCriterionErrorMsg(
    const std::string & aName
  ) const
  {
    std::string tMsg = std::string("ERROR: Criterion parameter list with name '")
      + aName + "' is not defined. " + "Defined criterion parameter lists are: ";
    for(const auto& tPair : mCriterionEvaluators)
    {
      tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    tSubMsg += ". The parameter list name and criterion ('Functions') arguments must match.";
    return tSubMsg;
  }

  void 
  updateDatabase(
    Plato::Database & aDatabase
  )
  {
    const Plato::OrdinalType tCycleIndex = aDatabase.scalar("cycle index");
    auto tMyDisplacement = Kokkos::subview(mDisplacements,tCycleIndex,Kokkos::ALL());
    aDatabase.vector("states", tMyDisplacement);
    auto tMyTemperature = Kokkos::subview(mTemperatures,tCycleIndex,Kokkos::ALL());
    aDatabase.vector("node states", tMyTemperature);
  }

  void 
  initializeCriterionEvaluators(
    Teuchos::ParameterList & aParamList
  )
  {
    if(aParamList.isSublist("Criteria"))
    {
      Plato::Elliptic::FactoryCriterionEvaluator<PhysicsType> tCriterionFactory;
      auto tCriteriaParams = aParamList.sublist("Criteria");
      for(Teuchos::ParameterList::ConstIterator tIndex=tCriteriaParams.begin(); tIndex!=tCriteriaParams.end(); ++tIndex)
      {
        const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
        std::string tCriterionName = tCriteriaParams.name(tIndex);
        TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
          " Parameter in ('Criteria') block not valid. Expect parameter lists only.");
        auto tCriterion = tCriterionFactory.create(mSpatialModel,mDataMap,aParamList,tCriterionName);
        if( tCriterion != nullptr )
        {
          mCriterionEvaluators[tCriterionName] = tCriterion;
        }
      }
    }
  }

  void 
  initializeThermalResidualEvaluators(
    Teuchos::ParameterList & aParamList
  )
  {
    auto tParamList = aParamList.sublist(mTypePDE);
    if(tParamList.isSublist("Thermal Residual") == false){
      auto tErrorMsg = std::string("ERROR: ('Thermal Residual') parameter list is not defined'");
      ANALYZE_THROWERR(tErrorMsg)
    }
    auto tMechResParamList = tParamList.sublist("Thermal Residual");
    // only linear thermal physics are supported/implemented
    mThermalResidualType = Plato::Elliptic::residual_t::LINEAR_THERMAL;
    mResidualEvaluators[mThermalResidualType] = 
      std::make_shared<Plato::Elliptic::VectorFunction<ThermalPhysicsType>>(mTypePDE,mSpatialModel,mDataMap,aParamList);
  }

  void 
  initializeMechanicalResidualEvaluators(
    Teuchos::ParameterList & aParamList
  )
  {
    auto tParamList = aParamList.sublist(mTypePDE);
    if(tParamList.isSublist("Mechanical Residual") == false){
      auto tErrorMsg = std::string("ERROR: ('Mechanical Residual') parameter list is not defined'");
      ANALYZE_THROWERR(tErrorMsg)
    }
    Plato::Elliptic::ResidualEnum tS2E;
    auto tMechResParamList = tParamList.sublist("Mechanical Residual");
    auto tResponse = tMechResParamList.get<std::string>("Response","linear");
    auto tResidualStringType = tResponse + " thermomechanical";
    mMechanicalResidualType = tS2E.get(tResidualStringType);
    mResidualEvaluators[mMechanicalResidualType] = 
      std::make_shared<Plato::Elliptic::VectorFunction<PhysicsType>>(mTypePDE,mSpatialModel,mDataMap,aParamList);
  }

  void 
  initializeSolvers(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aParamList,
    Comm::Machine          & aMachine
  )
  {
    LinearSystemType tSystemType = LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE;
    Plato::SolverFactory tSolverFactory(aParamList.sublist("Linear Solver"), tSystemType);
    for(auto& tPair : mResidualEvaluators)
    {
      mLinearSolvers[tPair.first] = 
        tSolverFactory.create(aMesh->NumNodes(), aMachine, tPair.second->numStateDofsPerNode(), nullptr);
    }
  }

  void 
  readEssentialBoundaryConditions(
    Teuchos::ParameterList & aParamList
  )
  {
    for(auto& tPair : mResidualEvaluators)
    {
      if( this->isThermalPhysics(tPair.second) ){
        this->readThermalEssentialBoundaryConditions(tPair.second->type(),aParamList);
      }
      else
      if( this->isMechanicalPhysics(tPair.second) ){
        this->readMechanicalEssentialBoundaryConditions(tPair.second->type(),aParamList);
      }
    }
  }

  void 
  readMechanicalEssentialBoundaryConditions(
    Plato::Elliptic::residual_t & aResidualType,
    Teuchos::ParameterList      & aParamList
  )
  {
    if(aParamList.isSublist("Mechanical Essential Boundary Conditions") == false)
    { 
      auto tErrorMsg = std::string("ERROR: Parameter list ('Mechanical Essential Boundary Conditions') ") + 
        "is not defined in the input deck";
      ANALYZE_THROWERR(tErrorMsg) 
    }
    Plato::EssentialBCs<PhysicsType> tMechanicalEBCs(
      aParamList.sublist("Mechanical Essential Boundary Conditions", false), mSpatialModel.Mesh
    );

    Plato::ScalarVector  tDirichletVals;
    Plato::OrdinalVector tDirichletDofs; 
    tMechanicalEBCs.get(tDirichletDofs,tDirichletVals);
    mDirichletDofs[aResidualType] = tDirichletDofs;
    mDirichletVals[aResidualType] = tDirichletVals;
    
    if(aParamList.isType<bool>("Weak Essential Boundary Conditions"))
    { mWeakEBCs = aParamList.get<bool>("Weak Essential Boundary Conditions",false); }
  }

  void 
  readThermalEssentialBoundaryConditions(
    Plato::Elliptic::residual_t & aResidualType,
    Teuchos::ParameterList      & aParamList
  )
  {
    if(aParamList.isSublist("Thermal Essential Boundary Conditions") == false)
    { 
      auto tErrorMsg = std::string("ERROR: Parameter list ('Thermal Essential Boundary Conditions') ") + 
        "is not defined in the input deck";
      ANALYZE_THROWERR(tErrorMsg) 
    }
    Plato::EssentialBCs<PhysicsType> tThermalEBCs(
      aParamList.sublist("Thermal Essential Boundary Conditions", false), mSpatialModel.Mesh
    );
    
    Plato::ScalarVector  tDirichletVals;
    Plato::OrdinalVector tDirichletDofs; 
    tThermalEBCs.get(tDirichletDofs,tDirichletVals);
    mDirichletDofs[aResidualType] = tDirichletDofs;
    mDirichletVals[aResidualType] = tDirichletVals;
    
    if(aParamList.isType<bool>("Weak Essential Boundary Conditions"))
    { mWeakEBCs = aParamList.get<bool>("Weak Essential Boundary Conditions",false); }
  }

};

} // namespace Elliptic

} // namespace Plato


namespace ThermoelastostaticTotalLagrangianTests
{

Teuchos::RCP<Teuchos::ParameterList> tGenericParamList = Teuchos::getParametersFromXmlString(
"<ParameterList name='Plato Problem'>                                                                      \n"
  "<ParameterList name='Spatial Model'>                                                                    \n"
    "<ParameterList name='Domains'>                                                                        \n"
      "<ParameterList name='Design Volume'>                                                                \n"
        "<Parameter name='Element Block' type='string' value='body'/>                                      \n"
        "<Parameter name='Material Model' type='string' value='Unobtainium'/>                              \n"
      "</ParameterList>                                                                                    \n"
    "</ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<Parameter name='PDE Constraint' type='string' value='Elliptic'/>                                       \n"
  "<Parameter name='Physics' type='string' value='Thermomechanical'/>                                      \n"
  "<ParameterList name='Elliptic'>                                                                         \n"
  "  <ParameterList name='Mechanical Residual'>                                                            \n"
  "    <Parameter name='Response' type='string' value='Nonlinear'/>                                        \n"
  "    <ParameterList name='Penalty Function'>                                                             \n"
  "      <Parameter name='Exponent' type='double' value='1.0'/>                                            \n"
  "      <Parameter name='Minimum Value' type='double' value='0.0'/>                                       \n"
  "      <Parameter name='Type' type='string' value='SIMP'/>                                               \n"
  "    </ParameterList>                                                                                    \n"
  "  </ParameterList>                                                                                      \n"
  "  <ParameterList name='Thermal Residual'>                                                               \n"
  "    <Parameter name='Response' type='string' value='Linear'/>                                           \n"
  "    <ParameterList name='Penalty Function'>                                                             \n"
  "      <Parameter name='Exponent' type='double' value='1.0'/>                                            \n"
  "      <Parameter name='Minimum Value' type='double' value='0.0'/>                                       \n"
  "      <Parameter name='Type' type='string' value='SIMP'/>                                               \n"
  "    </ParameterList>                                                                                    \n"
  "  </ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<ParameterList name='Material Models'>                                                                  \n"
    "<ParameterList name='Unobtainium'>                                                                    \n"
      "<ParameterList name='Thermal Conduction'>                                                           \n"
        "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>                              \n"
        "<Parameter  name='Reference Temperature' type='double' value='1.0'/>                              \n"
        "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>                              \n"
      "</ParameterList>                                                                                    \n"
      "<ParameterList name='Hyperelastic Kirchhoff'>                                                       \n"
        "<Parameter  name='Youngs Modulus'  type='double'  value='1.5'/>                                   \n"
        "<Parameter  name='Poissons Ratio'  type='double'  value='0.35'/>                                  \n"
      "</ParameterList>                                                                                    \n"
    "</ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<ParameterList name='Criteria'>                                                                         \n"
  "  <ParameterList name='Objective'>                                                                      \n"
  "    <Parameter name='Type' type='string' value='Weighted Sum'/>                                         \n"
  "    <Parameter name='Functions' type='Array(string)' value='{My Thermal Energy,My Mechanical Energy}'/> \n"
  "    <Parameter name='Weights' type='Array(double)' value='{1.0, 1.0}'/>                                 \n"
  "  </ParameterList>                                                                                      \n"
  "  <ParameterList name='My Thermal Energy'>                                                              \n"
  "    <Parameter name='Type'                 type='string' value='Scalar Function'/>                      \n"
  "    <Parameter name='Scalar Function Type' type='string' value='Internal Thermal Energy'/>              \n"
  "  </ParameterList>                                                                                      \n"
  "  <ParameterList name='My Mechanical Energy'>                                                           \n"
  "    <Parameter name='Type'                 type='string' value='Scalar Function'/>                      \n"
  "    <Parameter name='Scalar Function Type' type='string' value='Kirchhoff Energy Potential'/>           \n"
  "  </ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
"</ParameterList>                                                                                          \n"
); 

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, tComputeThermalDefGrad )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                             \n"
    "<ParameterList name='Material Models'>                                         \n"
      "<ParameterList name='Unobtainium'>                                           \n"
        "<ParameterList name='Thermal Conduction'>                                  \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>     \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>     \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>     \n"
        "</ParameterList>                                                           \n"
      "</ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  "</ParameterList>                                                                 \n"
  ); 
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  // create temperature workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temps", tNumNodes);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill node state",
    Kokkos::RangePolicy<>(0, tNumNodes), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<NodeStateT> tTempWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tTemp, tTempWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute thermal deformation gradient
  Plato::ThermalDeformationGradient<Residual> tComputeThermalDeformationGradient("Unobtainium",*tParamList);
  Plato::InterpolateFromNodal<ElementType,ElementType::mNumNodeStatePerNode> tInterpolateFromNodal;
  Kokkos::parallel_for("compute thermal deformation gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    // interpolate temperature from nodes to integration points
    NodeStateT tTemperature = tInterpolateFromNodal(iCellOrdinal,tBasisValues,tTempWS);
    // compute thermal deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> tTempDefGrad;
    tComputeThermalDeformationGradient(tTemperature,tTempDefGrad);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tTempDefGrad(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, tComputeThermoElasticDefGrad )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create thermal gradient workset
  std::vector<std::vector<Plato::Scalar>> tTempDefGrad = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<NodeStateT> tTempDefGradWS("thermal deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostTempDefGradWS = Kokkos::create_mirror(tTempDefGradWS);
  Kokkos::deep_copy(tHostTempDefGradWS, tTempDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostTempDefGradWS(tCell,tDimI,tDimJ) = tTempDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tTempDefGradWS,tHostTempDefGradWS);
  // create mechanical gradient workset
  std::vector<std::vector<Plato::Scalar>> tMechDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  Plato::ScalarArray3DT<StrainT> tMechDefGradWS("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostMechDefGradWS = Kokkos::create_mirror(tMechDefGradWS);
  Kokkos::deep_copy(tHostMechDefGradWS,tMechDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostMechDefGradWS(tCell,tDimI,tDimJ) = tMechDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tMechDefGradWS,tHostMechDefGradWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // compute thermo-elastic deformation gradient
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::ThermoElasticDeformationGradient<Residual> tComputeThermoElasticDeformationGradient;
  Kokkos::parallel_for("compute thermo-elastic deformation gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
      tCellMechDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellMechDefGrad(tDimI,tDimJ) = tMechDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> 
      tCellTempDefGrad(NodeStateT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellTempDefGrad(tDimI,tDimJ) = tTempDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> 
      tTMechDefGrad(ResultT(0.));
    tComputeThermoElasticDeformationGradient(tCellTempDefGrad,tCellMechDefGrad,tTMechDefGrad);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tTMechDefGrad(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.72333333,0.10333333,0.20666667,0.62}, {0.72333333,0.10333333,0.20666667,0.62} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, KineticPullBackOperation_1 )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create mechanical deformation gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<StrainT> tDefGradient("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostDefGradWS = Kokkos::create_mirror(tDefGradient);
  Kokkos::deep_copy(tHostDefGradWS, tDefGradient);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostDefGradWS(tCell,tDimI,tDimJ) = tDataDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tDefGradient,tHostDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::KineticPullBackOperation<Residual> tApplyKineticPullBackOperation;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tDefConfig2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tDefConfig2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tCellDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellDefGrad(tDimI,tDimJ) = tDefGradient(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tUndefConfig2PKS(ResultT(0.));
    tApplyKineticPullBackOperation(tCellDefGrad,tDefConfig2PKS,tUndefConfig2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tUndefConfig2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.9328371,-0.1743207,-0.1743207,0.9782719}, {0.9328371,-0.1743207,-0.1743207,0.9782719} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, KineticPullBackOperation_2 )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create thermal gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataTempDefGrad = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<NodeStateT> tTempDefGradWS("thermal deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostTempDefGradWS = Kokkos::create_mirror(tTempDefGradWS);
  Kokkos::deep_copy(tHostTempDefGradWS, tTempDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostTempDefGradWS(tCell,tDimI,tDimJ) = tDataTempDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tTempDefGradWS,tHostTempDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::KineticPullBackOperation<Residual> tApplyKineticPullBackOperation;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tDefConfig2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tDefConfig2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> 
      tCellTempDefGrad(NodeStateT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellTempDefGrad(tDimI,tDimJ) = tTempDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tUndefConfig2PKS(ResultT(0.));
    tApplyKineticPullBackOperation(tCellTempDefGrad,tDefConfig2PKS,tUndefConfig2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tUndefConfig2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, NominalStressTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create mechanical deformation gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<StrainT> tDefGradient("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostDefGradWS = Kokkos::create_mirror(tDefGradient);
  Kokkos::deep_copy(tHostDefGradWS, tDefGradient);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostDefGradWS(tCell,tDimI,tDimJ) = tDataDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tDefGradient,tHostDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::NominalStressTensor<Residual> tComputeNominalStressTensor;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tCell2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCell2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tCellDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellDefGrad(tDimI,tDimJ) = tDefGradient(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tNominalStressTensor(ResultT(0.));
    tComputeNominalStressTensor(tCellDefGrad,tCell2PKS,tNominalStressTensor);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tNominalStressTensor(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.6049342,0.7802452,0.5679006,1.1555556}, {1.6049342,0.7802452,0.5679006,1.1555556} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, get_cell_2PKS )
{
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  // create second piola-kirchhoff stress data
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tNumCells = 2;
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  Plato::ScalarArray4D t2PKS_WS("second piola-kirchhoff stress",tNumCells,/*num_intg_pts=*/1,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,0,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3D tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get cell 2PKS
  auto tNumPoints = ElementType::mNumGaussPoints;
  Kokkos::parallel_for("get cell 2PKS", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {

    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims> tCell2PKS(0.);
    Plato::Elliptic::get_cell_2PKS(iCellOrdinal,iGpOrdinal,t2PKS_WS,tCell2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tCell2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, Residual )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using ElementType  = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using ResidualEval = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT    = typename ResidualEval::StateScalarType;
  using ResultT      = typename ResidualEval::ResultScalarType;
  using ConfigT      = typename ResidualEval::ConfigScalarType;
  using NodeStateT   = typename ResidualEval::NodeStateScalarType;
  using StrainT      = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create displacement workset
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create temperature workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarVector tTemp("Temps", tNumVerts);
  Plato::blas1::fill(1., tTemp);
  Kokkos::parallel_for("fill temperature",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create control workset
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tDatabase.vector("controls",tControl);
  // create workset database
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<ResidualEval> tWorksetBuilder(tWorksetFuncs);
  tWorksetBuilder.build(tOnlyDomainDefined, tDatabase, tWorkSets);
  auto tNumCells = tMesh->NumElements();
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultT> > >
    ( Plato::ScalarMultiVectorT<ResultT>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  tWorkSets.set("result",tResultWS);
  // evaluate residualeval
  Plato::Elliptic::ResidualThermoElastoStaticTotalLagrangian<ResidualEval> 
    tResidual(tOnlyDomainDefined,tDataMap,*tGenericParamList);
  tResidual.evaluate(tWorkSets,/*cycle=*/0.);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { 
      {-1.6049390,-0.567902,0.824691,-0.587654,0.78024773,1.155556}, 
      {-0.6827165,-1.011111,1.404321, 0.496914,-0.7216045,0.514197} 
    };
  auto tHostResultsWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultsWS, tResultWS->mData);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      TEST_FLOATING_EQUALITY(tGold[tCell][tDof],tHostResultsWS(tCell,tDof),tTolerance);
    }
  }
}

} 