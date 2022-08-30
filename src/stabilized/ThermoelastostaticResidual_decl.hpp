#pragma once

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "PlatoTypes.hpp"
#include "ApplyWeighting.hpp"
#include "LinearThermoelasticMaterial.hpp"
#include "stabilized/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermoelastostaticResidual :
        public EvaluationType::ElementType,
        public Plato::Stabilized::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumSpatialDims;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;

    static constexpr int NMechDims  = mNumSpatialDims;
    static constexpr int NPressDims = 1;
    static constexpr int NThrmDims  = 1;

    static constexpr int MDofOffset = 0;
    static constexpr int PDofOffset = mNumSpatialDims;
    static constexpr int TDofOffset = mNumSpatialDims+1;

    using FunctionBaseType = Plato::Stabilized::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyTensorWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyVectorWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, /*num_dofs=*/1,  IndicatorFunctionType> mApplyScalarWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;

    std::shared_ptr<Plato::NaturalBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>> mBoundaryFluxes;

    Teuchos::RCP<Plato::LinearThermoelasticMaterial<mNumSpatialDims>> mMaterialModel;

public:
    /**************************************************************************/
    ThermoelastostaticResidual(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap& aDataMap,
              Teuchos::ParameterList& aProblemParams,
              Teuchos::ParameterList& aPenaltyParams
    );

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override;

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT<StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT<NodeStateScalarType> & aPGradWS,
        const Plato::ScalarMultiVectorT<ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT<ConfigScalarType>        & aConfigWS,
              Plato::ScalarMultiVectorT<ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
    /**************************************************************************/

    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                            & aSpatialModel,
        const Plato::ScalarMultiVectorT<StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT<NodeStateScalarType> & aPGradWS,
        const Plato::ScalarMultiVectorT<ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT<ConfigScalarType>        & aConfigWS,
              Plato::ScalarMultiVectorT<ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
    /**************************************************************************/
};
// class ThermoelastostaticResidual

} // namespace Stabilized
} // namespace Plato
