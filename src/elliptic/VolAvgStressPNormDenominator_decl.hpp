#pragma once

#include "TensorPNorm.hpp"
#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class VolAvgStressPNormDenominator : 
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, /*number of terms=*/1, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

    std::string mSpatialWeightFunction = "1.0";

  public:
    /**************************************************************************/
    VolAvgStressPNormDenominator(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    );

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;

    /**************************************************************************/
    void
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar);

    /**************************************************************************/
    void
    postEvaluate(Plato::Scalar& resultValue);
};
// class VolAvgStressPNormDenominator

} // namespace Elliptic

} // namespace Plato
