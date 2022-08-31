#pragma once

#include <memory>

#include "FadTypes.hpp"
#include "PlatoTypes.hpp"
#include "CellForcing.hpp"
#include "ProjectToNode.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "stabilized/PressureGradient.hpp"

namespace Plato
{

namespace Stabilized
{

    /***************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input XML data, i.e. parameter list
    *******************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    PressureGradientProjectionResidual<EvaluationType, IndicatorFunctionType>::
    initialize(Teuchos::ParameterList &aProblemParams)
    {
        mPressureScaling = 1.0;
        if (aProblemParams.isSublist("Material Models"))
        {
            Teuchos::ParameterList& tMaterialsInputs = aProblemParams.sublist("Material Models");
            mPressureScaling =      tMaterialsInputs.get<Plato::Scalar>("Pressure Scaling", 1.0);
            Teuchos::ParameterList& tMaterialInputs = tMaterialsInputs.sublist(mSpatialDomain.getMaterialName());
        }
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aDataMap output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    PressureGradientProjectionResidual<EvaluationType, IndicatorFunctionType>::
    PressureGradientProjectionResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyVectorWeighting (mIndicatorFunction),
        mPressureScaling      (1.0)
    {
        this->initialize(aProblemParams);
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Solutions
    PressureGradientProjectionResidual<EvaluationType, IndicatorFunctionType>::
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
      Plato::Solutions tSolutionsOutput(aSolutions.physics(), aSolutions.pde());
      return tSolutionsOutput;
    }

    /******************************************************************************//**
     * \brief Evaluate stabilized elastostatics residual
     * \param [in] aNodalPGradWS pressure gradient workset on H^1(\Omega)
     * \param [in] aPressureWS pressure gradient workset on H^1(\Omega)
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    PressureGradientProjectionResidual<EvaluationType, IndicatorFunctionType>::
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aNodalPGradWS,
        const Plato::ScalarMultiVectorT <NodeStateScalarType> & aPressureWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarMultiVectorT <ResultScalarType>    & aResultWS,
              Plato::Scalar                                     aTimeStep
    ) const
    {
        auto tNumCells = mSpatialDomain.numCells();

        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
        Plato::PressureGradient<ElementType> tComputePressureGradient(mPressureScaling);
        Plato::InterpolateFromNodal<ElementType, mNumSpatialDims, 0, mNumSpatialDims> tInterpolatePressGradFromNodal;
        Plato::ProjectToNode<ElementType> tProjectPressGradToNodal;

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto& tApplyVectorWeighting = mApplyVectorWeighting;
        Kokkos::parallel_for("Projected pressure gradient residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            // compute gradient operator and cell volume
            //
            tComputeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
            tVolume *= tCubWeights(iGpOrdinal);

            // compute pressure gradient
            //
            Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tPressureGrad (0.0);
            tComputePressureGradient(iCellOrdinal, tPressureGrad, aPressureWS, tGradient);

            // interpolate projected pressure gradient from nodes
            //
            Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tProjectedPGrad (0.0);
            tInterpolatePressGradFromNodal (iCellOrdinal, tBasisValues, aNodalPGradWS, tProjectedPGrad);

            // apply weighting
            //
            tApplyVectorWeighting(iCellOrdinal, aControlWS, tBasisValues, tPressureGrad);
            tApplyVectorWeighting(iCellOrdinal, aControlWS, tBasisValues, tProjectedPGrad);

            // project pressure gradient to nodes
            //
            tProjectPressGradToNodal(iCellOrdinal, tVolume, tBasisValues, tProjectedPGrad, aResultWS);
            tProjectPressGradToNodal(iCellOrdinal, tVolume, tBasisValues, tPressureGrad, aResultWS, /*scale=*/-1.0);

        });
    }

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aState     global state variables
     * \param [in] aControl   control variables, e.g. design variables
     * \param [in] aTimeStep  pseudo time step
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    PressureGradientProjectionResidual<EvaluationType, IndicatorFunctionType>::
    updateProblem(
        const Plato::ScalarMultiVector & aState,
        const Plato::ScalarVector      & aControl,
              Plato::Scalar              aTimeStep)
    {
        mApplyVectorWeighting.update();
    }
} // namespace Stabilized
} // namespace Plato
