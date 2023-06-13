#pragma once

#include "TensorPNorm.hpp"
#include "ApplyWeighting.hpp"
#include "base/CriterionBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class CriterionVolAvgStressPNormDenominator : public Plato::CriterionBase
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;

    using Plato::CriterionBase::mSpatialDomain;
    using Plato::CriterionBase::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    
    IndicatorFunctionType mIndicatorFunction;
    std::string mSpatialWeightFunction = "1.0";
    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;
    Plato::ApplyWeighting<mNumNodesPerCell, /*number of terms=*/1, IndicatorFunctionType> mApplyWeighting;

public:
  CriterionVolAvgStressPNormDenominator(
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

}; // class CriterionVolAvgStressPNormDenominator

} // namespace Elliptic

} // namespace Plato
