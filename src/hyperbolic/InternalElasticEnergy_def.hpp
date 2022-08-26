#pragma once

#include "hyperbolic/InternalElasticEnergy_decl.hpp"

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"

#include "GradientMatrix.hpp"

#include "SmallStrain.hpp"
#include "LinearStress.hpp"

#include "ScalarProduct.hpp"

namespace Plato
{

namespace Hyperbolic
{

    template<typename EvaluationType, typename IndicatorFunctionType>
    InternalElasticEnergy<EvaluationType, IndicatorFunctionType>::InternalElasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    {
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    InternalElasticEnergy<EvaluationType, IndicatorFunctionType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>       & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType>    & aStateDot,
        const Plato::ScalarMultiVectorT <StateDotDotScalarType> & aStateDotDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep
    ) const
    {
        using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

        auto tNumCells = mSpatialDomain.numCells();

        Plato::ComputeGradientMatrix<ElementType> computeGradient;
        Plato::SmallStrain<ElementType>           computeVoigtStrain;
        Plato::ScalarProduct<mNumVoigtTerms>      computeScalarProduct;

        Plato::LinearStress<EvaluationType, ElementType> computeVoigtStress(mMaterialModel);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto& applyWeighting = mApplyWeighting;
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrain(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

            computeVoigtStrain(iCellOrdinal, tStrain, aState, tGradient);

            computeVoigtStress(tStress, tStrain);

            tVolume *= tCubWeights(iGpOrdinal);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            applyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

            computeScalarProduct(iCellOrdinal, aResult, tStress, tStrain, tVolume);
        });
    }
} 

} 
