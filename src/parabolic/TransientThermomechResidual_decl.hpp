#pragma once

#include "NaturalBCs.hpp"
#include "MaterialModel.hpp"
#include "ApplyWeighting.hpp"
#include "parabolic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TransientThermomechResidual : 
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
    using ElementType::mNumVoigtTerms;

    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    static constexpr int NThrmDims = 1;
    static constexpr int NMechDims = SpaceDim;

    static constexpr int TDofOffset = SpaceDim;
    static constexpr int MDofOffset = 0;

    using FunctionBaseType = Plato::Parabolic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;
    using FunctionBaseType::mDofDotNames;

    using StateScalarType    = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using ControlScalarType  = typename EvaluationType::ControlScalarType;
    using ConfigScalarType   = typename EvaluationType::ConfigScalarType;
    using ResultScalarType   = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyFluxWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, NThrmDims,       IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::NaturalBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>> mBoundaryFluxes;

    Teuchos::RCP<Plato::MaterialModel<EvaluationType>> mMaterialModel;
    Teuchos::RCP<Plato::MaterialModel<EvaluationType>> mThermalMassMaterialModel;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    TransientThermomechResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
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
              Plato::Scalar aTimeStep = 0.0) const override;

    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                             & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0) const override;
    /**************************************************************************/
};

} // namespace Parabolic

} // namespace Plato
