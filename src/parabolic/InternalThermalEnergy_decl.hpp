#pragma once

#include "ApplyWeighting.hpp"
#include "ThermalConductivityMaterial.hpp"
#include "parabolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************//**
 * \brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermalEnergy : 
  public EvaluationType::ElementType,
  public Plato::Parabolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType    = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using ControlScalarType  = typename EvaluationType::ControlScalarType;
    using ConfigScalarType   = typename EvaluationType::ConfigScalarType;
    using ResultScalarType   = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Parabolic::AbstractScalarFunction<EvaluationType>;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mThermalConductivityMaterialModel;

  public:
    /**************************************************************************/
    InternalThermalEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    );

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>    & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType> & aStateDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>  & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>   & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>   & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
};
// class InternalThermalEnergy

} // namespace Parabolic

} // namespace Plato
