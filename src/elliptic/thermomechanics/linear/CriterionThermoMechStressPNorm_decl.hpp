#pragma once

#include "TensorPNorm.hpp"
#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "ThermoelasticMaterial.hpp"

#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
class CriterionThermoMechStressPNorm : public Plato::CriterionBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of stress-strain terms
  static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;
  /// @brief number of integration points
  static constexpr auto mNumGaussPoints  = ElementType::mNumGaussPoints;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief temperaturedegree of freedom offset
  static constexpr int TDofOffset = mNumSpatialDims;

  using FunctionBaseType = typename Plato::CriterionBase;
  using FunctionBaseType::mSpatialDomain;
  using FunctionBaseType::mDataMap;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using GradScalarType    = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

  IndicatorFunctionType mIndicatorFunction;
  Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
  /// @brief interface for material constitutive model
  Teuchos::RCP<Plato::MaterialModel<EvaluationType>> mMaterialModel;

  Teuchos::RCP<TensorNormBase<mNumVoigtTerms, EvaluationType>> mNorm;

  std::string mFuncString = "1.0";

public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       ouput database
  /// @param [in] aProblemParams input problem parameters
  /// @param [in] aPenaltyParams input penalty model parameters
  /// @param [in] aFunctionName  criterion parameter list name
  CriterionThermoMechStressPNorm(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap, 
          Teuchos::ParameterList & aProblemParams, 
          Teuchos::ParameterList & aPenaltyParams,
    const std::string            & aFunctionName
  );

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn evaluateConditional
  /// @brief evaluate thermo-mechanical stress p-norm criterion
  /// @param [in,out] aWorkSets function domain and range workset database
  /// @param [in]     aCycle    scalar 
  void
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;

  void
  postEvaluate( 
    Plato::ScalarVector resultVector,
    Plato::Scalar       resultScalar
  );

  void
  postEvaluate( 
    Plato::Scalar& resultValue 
  );

};
// class CriterionThermoMechStressPNorm

} // namespace Elliptic

} // namespace Plato

