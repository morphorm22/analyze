#pragma once

#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "GradientMatrix.hpp"
#include "VonMisesYieldFunction.hpp"

namespace Plato
{

    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    template<typename EvaluationType>
    VonMisesLocalMeasure<EvaluationType>::
    VonMisesLocalMeasure(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) : 
        AbstractLocalMeasure<EvaluationType>(aSpatialDomain, aDataMap, aInputParams, aName)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aInputParams);
        auto tMaterialModel = tMaterialModelFactory.create(tMaterialName);
        mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aCellStiffMatrix stiffness matrix
     * \param [in] aName local measure name
     **********************************************************************************/
    template<typename EvaluationType>
    VonMisesLocalMeasure<EvaluationType>::
    VonMisesLocalMeasure(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap,
        const MatrixType           & aCellStiffMatrix,
        const std::string            aName
    ) :
        AbstractLocalMeasure<EvaluationType>(aSpatialDomain, aDataMap, aName)
    {
        mCellStiffMatrix = aCellStiffMatrix;
    }

    /******************************************************************************//**
     * \brief Evaluate vonmises local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    template<typename EvaluationType>
    void
    VonMisesLocalMeasure<EvaluationType>::
    operator()(
        const Plato::ScalarMultiVectorT<StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT<ControlT> & aControlWS,
        const Plato::ScalarArray3DT<ConfigT>      & aConfigWS,
              Plato::ScalarVectorT<ResultT>       & aResultWS
    )
    {
        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = aResultWS.size();

        Plato::SmallStrain<ElementType> tComputeCauchyStrain;
        Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;
        Plato::ComputeGradientMatrix<ElementType> tComputeGradientMatrix;

        Plato::LinearStress<EvaluationType, ElementType> tComputeCauchyStress(mCellStiffMatrix);

        Plato::ScalarVectorT<ConfigT> tCellVolume("volume", tNumCells);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigT tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigT> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainT> tStrain(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultT> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            tComputeGradientMatrix(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
            tComputeCauchyStrain(iCellOrdinal, tStrain, aStateWS, tGradient);
            tComputeCauchyStress(tStress, tStrain);

            ResultT tResult(0);
            tComputeVonMises(iCellOrdinal, tStress, tResult);
            Kokkos::atomic_add(&aResultWS(iCellOrdinal), tResult*tVolume);
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            aResultWS(iCellOrdinal) /= tCellVolume(iCellOrdinal);
        });
    }
}
//namespace Plato
