#pragma once

#include "TensorPNorm.hpp"
#include "ApplyWeighting.hpp"
#include "LinearElectroelasticMaterial.hpp"

#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
class EMStressPNorm : public Plato::CriterionBase
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

  using FunctionBaseType = typename Plato::CriterionBase;
  using FunctionBaseType::mSpatialDomain;
  using FunctionBaseType::mDataMap;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using GradScalarType    = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

  Teuchos::RCP<Plato::LinearElectroelasticMaterial<mNumSpatialDims>> mMaterialModel;

  IndicatorFunctionType mIndicatorFunction;
  Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

  Teuchos::RCP<TensorNormBase<mNumVoigtTerms, EvaluationType>> mNorm;

  std::string mFuncString = "1.0";

public:
  EMStressPNorm(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap, 
            Teuchos::ParameterList & aProblemParams, 
            Teuchos::ParameterList & aPenaltyParams,
      const std::string            & aFunctionName
  );

  /// @brief returns true if criterion is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn evaluateConditional
  /// @brief evaluate criterion
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
// class EMStressPNorm

} // namespace Elliptic

} // namespace Plato
