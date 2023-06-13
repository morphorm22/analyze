#pragma once

#include "TensorPNorm.hpp"
#include "ApplyWeighting.hpp"
#include "base/CriterionBase.hpp"
#include "ElasticModelFactory.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class CriterionStressPNorm : public Plato::CriterionBase
/******************************************************************************/
{
private:
  using ElementType = typename EvaluationType::ElementType;

  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;

  using Plato::CriterionBase::mSpatialDomain;
  using Plato::CriterionBase::mDataMap;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;

  std::string mFuncString = "1.0";
  Plato::ScalarMultiVector mFxnValues;
  
  IndicatorFunctionType mIndicatorFunction;
  
  Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;
  Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;
  Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;


public:
  CriterionStressPNorm(
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
// class CriterionStressPNorm

} // namespace Elliptic

} // namespace Plato
