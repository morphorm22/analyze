#pragma once

#include "FadTypes.hpp"
#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename PenaltyFunctionType>
class Volume :
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;
 
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, /*num_weighted_terms=*/ 1, PenaltyFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    Volume(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    );

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT<StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
};
// class Volume

} // namespace Elliptic

} // namespace Plato
