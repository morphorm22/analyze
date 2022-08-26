#pragma once

#include "FadTypes.hpp"
#include "Kinetics.hpp"
#include "Kinematics.hpp"
#include "ScalarProduct.hpp"
#include "ProjectToNode.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "stabilized/MechanicsElement.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************//**
 * \brief Compute internal elastic energy criterion for stabilized form.
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    ElastostaticEnergy<EvaluationType, IndicatorFunctionType>::
    ElastostaticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap&          aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyTensorWeighting (mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    ElastostaticEnergy<EvaluationType, IndicatorFunctionType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarVectorT      <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix  <ElementType> tComputeGradient;
      Plato::Stabilized::Kinematics <ElementType> tKinematics;
      Plato::Stabilized::Kinetics   <ElementType> tKinetics(mMaterialModel);

      Plato::InterpolateFromNodal   <ElementType, mNumDofsPerNode, mPressDofOffset> tInterpolatePressureFromNodal;

      Plato::ScalarProduct<mNumVoigtTerms> tDeviatorScalarProduct;
      
      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& tApplyTensorWeighting = mApplyTensorWeighting;

      Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        ConfigScalarType tVolume(0.0);

        Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;


        // compute gradient operator and cell volume
        //
        auto tCubPoint = tCubPoints(iGpOrdinal);
        tComputeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
        tVolume *= tCubWeights(iGpOrdinal);

        // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
        //
        Plato::Array<mNumVoigtTerms, GradScalarType> tDGrad(0.0);
        Plato::Array<mNumSpatialDims, GradScalarType> tPGrad(0.0);
        tKinematics(iCellOrdinal, tDGrad, tPGrad, aStateWS, tGradient);

        // interpolate projected pressure to gauss point
        //
        ResultScalarType tPressure(0.0);
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        tInterpolatePressureFromNodal(iCellOrdinal, tBasisValues, aStateWS, tPressure);

        // compute the constitutive response
        //
        ResultScalarType tVolStrain(0.0);
        Plato::Array<mNumSpatialDims, ResultScalarType> tCellStab(0.0);
        Plato::Array<mNumSpatialDims, ResultScalarType> tProjectedPGrad(0.0);
        Plato::Array<mNumVoigtTerms, ResultScalarType> tDevStress(0.0);
        tKinetics(tVolume, tProjectedPGrad, tDGrad, tPGrad,
                  tPressure, tDevStress, tVolStrain, tCellStab);

        Plato::Array<mNumVoigtTerms, ResultScalarType> tTotStress(0.0);
        for( int i=0; i<mNumSpatialDims; i++)
        {
            tTotStress(i) = tDevStress(i) + tPressure;
        }

        // apply weighting
        //
        tApplyTensorWeighting (iCellOrdinal, aControlWS, tBasisValues, tTotStress);

        // compute element internal energy (inner product of strain and weighted stress)
        //
        tDeviatorScalarProduct(iCellOrdinal, aResultWS, tTotStress, tDGrad, tVolume);
      });
    }
} // namespace Stabilized
} // namespace Plato
