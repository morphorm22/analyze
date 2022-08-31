#pragma once

#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class IntermediateDensityPenalty :
    public EvaluationType::ElementType,
    public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mPenaltyAmplitude;

  public:
    /**************************************************************************/
    IntermediateDensityPenalty(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams,
              std::string              aFunctionName
    );

    /**************************************************************************
     * Unit testing constructor
     **************************************************************************/
    IntermediateDensityPenalty(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    );

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT<ResultScalarType>       & aResult,
              Plato::Scalar                                  aTimeStep = 0.0
    ) const ;
};
// class IntermediateDensityPenalty

}
// namespace Plato
