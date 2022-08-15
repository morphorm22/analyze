#pragma once

#include "TensorPNorm.hpp"
#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "hyperbolic/AbstractScalarFunction.hpp"
#include "hyperbolic/EvaluationTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

template<typename EvaluationType, typename IndicatorFunctionType>
class StressPNorm : 
  public EvaluationType::ElementType,
  public Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain; 
    using FunctionBaseType::mDataMap; 

    using StateScalarType       = typename EvaluationType::StateScalarType;
    using StateDotScalarType    = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

    std::string mFuncString = "1.0";
    
    Plato::ScalarMultiVector mFxnValues;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

  public:
    StressPNorm(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ); 

    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>       & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType>    & aStateDot,
        const Plato::ScalarMultiVectorT <StateDotDotScalarType> & aStateDotDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;

    void
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar);

    void
    postEvaluate( Plato::Scalar& resultValue );
}; 

} 

} 
