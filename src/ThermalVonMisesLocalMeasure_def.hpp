#pragma once

#include "FadTypes.hpp"
#include "TMKinematics.hpp"
#include "GradientMatrix.hpp"
#include "TMKineticsFactory.hpp"
#include "InterpolateFromNodal.hpp"
#include "VonMisesYieldFunction.hpp"

namespace Plato
{

    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    template<typename EvaluationType>
    ThermalVonMisesLocalMeasure<EvaluationType>::
    ThermalVonMisesLocalMeasure(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) : 
        AbstractLocalMeasure<EvaluationType>(aSpatialDomain, aDataMap, aInputParams, aName)
    {
        Plato::ThermoelasticModelFactory<mNumSpatialDims> tFactory(aInputParams);
        mMaterialModel = tFactory.create(mSpatialDomain.getMaterialName());
    }


    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    template<typename EvaluationType>
    ThermalVonMisesLocalMeasure<EvaluationType>::
    ~ThermalVonMisesLocalMeasure()
    {
    }

    /******************************************************************************//**
     * \brief Evaluate vonmises local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aDataMap map to stored data
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    template<typename EvaluationType>
    void
    ThermalVonMisesLocalMeasure<EvaluationType>::
    operator()(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS
    )
    {
        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = aResultWS.size();

        Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;

        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
        Plato::TMKinematics<ElementType>          tKinematics;

        Plato::TMKineticsFactory< EvaluationType, ElementType > tTMKineticsFactory;
        auto tTMKinetics = tTMKineticsFactory.create(mMaterialModel, mSpatialDomain, mDataMap);

        Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;

        Plato::ScalarVectorT<ConfigT> tCellVolume("volume", tNumCells);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Plato::ScalarArray3DT<ResultT> tStress("stress", tNumCells, tNumPoints, mNumVoigtTerms);
        Plato::ScalarArray3DT<ResultT> tFlux  ("flux",   tNumCells, tNumPoints, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrain("strain", tNumCells, tNumPoints, mNumVoigtTerms);
        Plato::ScalarArray3DT<StrainT> tTGrad ("tgrad",  tNumCells, tNumPoints, mNumSpatialDims);

        Plato::ScalarArray4DT<ConfigT> tGradient("gradient", tNumCells, tNumPoints, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarMultiVectorT<ConfigT> tVolume("volume", tNumCells, tNumPoints);
        Plato::ScalarMultiVectorT<StateT> tTemperature("temperature", tNumCells, tNumPoints);

        Kokkos::parallel_for("compute element state", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);

            tComputeGradient(iCellOrdinal, iGpOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);

            tVolume(iCellOrdinal, iGpOrdinal) *= tCubWeights(iGpOrdinal);

            // compute strain and electric field
            //
            tKinematics(iCellOrdinal, iGpOrdinal, tStrain, tTGrad, aStateWS, tGradient);

            // compute stress and electric displacement
            //
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tTemperature(iCellOrdinal, iGpOrdinal) = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aStateWS);
        });

        // compute element state
        (*tTMKinetics)(tStress, tFlux, tStrain, tTGrad, tTemperature, aControlWS);

        Kokkos::parallel_for("compute element state", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ResultT tResult(0);
            tComputeVonMises(iCellOrdinal, iGpOrdinal, tStress, tResult);
            Kokkos::atomic_add(&aResultWS(iCellOrdinal), tResult*tVolume(iCellOrdinal, iGpOrdinal));
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume(iCellOrdinal, iGpOrdinal));
        });

        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            aResultWS(iCellOrdinal) /= tCellVolume(iCellOrdinal);
        });

    }
}
//namespace Plato
