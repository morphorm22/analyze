/*
 *  CriterionPowerSurfaceDensityTwoPhase_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "MaterialModel.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/electrical/CurrentDensitySourceEvaluator.hpp"

namespace Plato
{

/// @class CriterionPowerSurfaceDensityTwoPhase
/// @brief power density surface (P) criterion for two-phase material use cases. the  
///   criterion is defined as: \f$P = \int_{\Omega}QV\, d\Omega\f$, where \f$Q\f$ is
///   the source term and \f$V\f$ is the electric potential
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class CriterionPowerSurfaceDensityTwoPhase : public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions 
  static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of degrees of freedom per node
  static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr int mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief number of nodes per cell
  static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief contains mesh and model information
  using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
  /// @brief output data map
  using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;
  /// @brief scalar types associated with the evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  /// @brief typename for base class
  using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
  /// @brief name of criterion parameter list in the input file
  std::string mCriterionFunctionName = "";
  /// @brief penalty exponent for material penalty model
  Plato::Scalar mPenaltyExponent = 3.0;
  /// @brief minimum value for the ersatz material density
  Plato::Scalar mMinErsatzMaterialValue = 0.0;
  /// @brief list of out-of-plane material thickness
  std::vector<Plato::Scalar> mOutofPlaneThickness;
  /// @brief current density evaluator 
  std::shared_ptr<Plato::CurrentDensitySourceEvaluator<EvaluationType>> mCurrentDensitySourceEvaluator;

public:
  /// @brief class constructor
  /// @param aSpatialDomain contains mesh and model information
  /// @param aDataMap       output data map
  /// @param aParamList     input problem parameters
  /// @param aFuncName      name of criterion parameter list
  CriterionPowerSurfaceDensityTwoPhase(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aParamList,
      const std::string            & aFuncName
  );

  /// @brief class destructor
  ~CriterionPowerSurfaceDensityTwoPhase();

  /// @fn evaluate
  /// @brief virtual function, overrides base class: evaluate criterion
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  2D result workset
  /// @param [in]     aCycle   scalar
  void 
  evaluate(
      const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType> & aControl,
      const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>  & aConfig,
            Plato::ScalarVectorT      <typename EvaluationType::ResultScalarType>  & aResult,
            Plato::Scalar                                                            aCycle = 1.0
  ) override;

  /// @fn evaluate_conditional
  /// @brief virtual function, overrides base class: evaluate criterion
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  2D result workset
  /// @param [in]     aCycle   scalar
  void
  evaluate_conditional(
      const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
      const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
            Plato::ScalarVectorT      <ResultScalarType>  & aResult,
            Plato::Scalar                                   aCycle = 1.0
  ) const override;

private:
  /// @fn initialize
  /// @brief initialize material constitutive model
  /// @param [in] aParamList input problem parameters
  void initialize(
    Teuchos::ParameterList & aParamList
  );

  /// @fn buildCurrentDensityFunction
  /// @brief build current density evaluator
  /// @param [in] aParamList input problem parameters
  void buildCurrentDensityFunction(
    Teuchos::ParameterList & aParamList
  );

  /// @fn setOutofPlaneThickness
  /// @brief iset list out-of-plane thicknesses
  /// @param [in] aMaterialModel material model interface
  void 
  setOutofPlaneThickness(
      Plato::MaterialModel<EvaluationType> & aMaterialModel
  );

};

}