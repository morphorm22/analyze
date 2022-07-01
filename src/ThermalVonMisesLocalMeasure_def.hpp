#pragma once

#include "FadTypes.hpp"
#include "TMKinetics.hpp"
#include "TMKinematics.hpp"
#include "GradientMatrix.hpp"
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
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) : 
        AbstractLocalMeasure<EvaluationType>(aSpatialDomain, aInputParams, aName)
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
        const Plato::ScalarMultiVectorT <StateT>  & aStateWS,
        const Plato::ScalarArray3DT     <ConfigT> & aConfigWS,
              Plato::ScalarVectorT      <ResultT> & aResultWS
    )
    {
        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = aResultWS.size();

        Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;

        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
        Plato::TMKinematics<ElementType>          tKinematics;
        Plato::TMKinetics<ElementType>            tKinetics(mMaterialModel);

        Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;

        Plato::ScalarVectorT<ConfigT> tCellVolume("volume", tNumCells);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute element state", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigT tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigT> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms,  StrainT> tStrain(0.0);
            Plato::Array<ElementType::mNumSpatialDims, StrainT> tTGrad (0.0);
            Plato::Array<ElementType::mNumVoigtTerms,  ResultT> tStress(0.0);
            Plato::Array<ElementType::mNumSpatialDims, ResultT> tFlux  (0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            tComputeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);

            tVolume *= tCubWeights(iGpOrdinal);

            // compute strain and electric field
            //
            tKinematics(iCellOrdinal, tStrain, tTGrad, aStateWS, tGradient);

            // compute stress and electric displacement
            //
            StateT tTemperature(0.0);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tInterpolateFromNodal(iCellOrdinal, tBasisValues, aStateWS, tTemperature);
            tKinetics(tStress, tFlux, tStrain, tTGrad, tTemperature);

            ResultT tResult(0);
            tComputeVonMises(iCellOrdinal, tStress, tResult);
            Kokkos::atomic_add(&aResultWS(iCellOrdinal), tResult*tVolume);
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal)
        {
            aResultWS(iCellOrdinal) /= tCellVolume(iCellOrdinal);
        });

    }
}
//namespace Plato
