#pragma once

#include "NaturalBCs.hpp"
#include "ApplyWeighting.hpp"
#include "ThermalMassMaterial.hpp"
#include "parabolic/EvaluationTypes.hpp"
#include "ThermalConductivityMaterial.hpp"
#include "parabolic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class HeatEquationResidual : 
  public EvaluationType::ElementType,
  public Plato::Parabolic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = Plato::Parabolic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;
    using FunctionBaseType::mDofDotNames;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using StateDotScalarType  = typename EvaluationType::StateDotScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyFluxWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumDofsPerNode, IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::NaturalBCs<ElementType, mNumDofsPerNode>> mBoundaryLoads;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mThermalMassMaterialModel;
    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mThermalConductivityMaterialModel;

  public:
    /**************************************************************************/
    HeatEquationResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & problemParams,
              Teuchos::ParameterList & penaltyParams
    );

    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override;

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT< StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;

    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                             & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
};
// class HeatEquationResidual

} // namespace Parabolic

} // namespace Plato
