#pragma once

#include "SmallStrain.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "elliptic/hatching/LinearStress.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

    /******************************************************************************//**
     * @brief Constructor
     * @param aSpatialDomain Plato Analyze spatial domain
     * @param aProblemParams input database for overall problem
     * @param aPenaltyParams input database for penalty function
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    InternalElasticEnergy<EvaluationType, IndicatorFunctionType>::
    InternalElasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        Plato::Elliptic::Hatching::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    {
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());
    }

    /******************************************************************************//**
     * @brief Evaluate internal elastic energy function
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    InternalElasticEnergy<EvaluationType, IndicatorFunctionType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarArray3DT     <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep
    ) const
    {
        using StrainScalarType = typename Plato::fad_type_t<ElementType, GlobalStateScalarType, ConfigScalarType>;
      
        auto tNumCells = mSpatialDomain.numCells();
      
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
        Plato::SmallStrain<ElementType>           tComputeVoigtStrainIncrement;
        Plato::ScalarProduct<mNumVoigtTerms>      tComputeScalarProduct;

        Plato::Elliptic::Hatching::LinearStress<ElementType> tComputeVoigtStress(mMaterialModel);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto tApplyWeighting  = mApplyWeighting;
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrainIncrement(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);
            tVolume *= tCubWeights(iGpOrdinal);

            // compute strain increment
            //
            tComputeVoigtStrainIncrement(iCellOrdinal, tStrainIncrement, aGlobalState, tGradient);

            // compute stress
            //
            tComputeVoigtStress(iCellOrdinal, iGpOrdinal, tStress, tStrainIncrement, aLocalState);

            // apply weighting
            //
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tApplyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
    
            // compute element internal energy (0.5 * inner product of total strain and weighted stress)
            //
            tComputeScalarProduct(iCellOrdinal, aResult, tStress, tStrainIncrement, tVolume, 0.5);

            Plato::Array<ElementType::mNumVoigtTerms, LocalStateScalarType> tLocalState;
            for(Plato::OrdinalType iVoigt=0; iVoigt<ElementType::mNumVoigtTerms; iVoigt++)
            {
                tLocalState(iVoigt) = aLocalState(iCellOrdinal, iGpOrdinal, iVoigt);
            }
            tComputeScalarProduct(iCellOrdinal, aResult, tStress, tLocalState, tVolume, 0.5);
        });
    }
} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
