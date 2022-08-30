#pragma once

#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************//**
 * \brief Compute internal elastic energy criterion for stabilized form.
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticEnergy : 
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNMechDims  = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNPressDims = 1;

    static constexpr Plato::OrdinalType mMDofOffset = 0;
    static constexpr Plato::OrdinalType mPressDofOffset = mNumSpatialDims;
    
    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyTensorWeighting;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

  public:
    /**************************************************************************/
    ElastostaticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap&          aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    );
    /**************************************************************************/

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarVectorT      <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
    /**************************************************************************/
};
// class InternalElasticEnergy

} // namespace Stabilized
} // namespace Plato
