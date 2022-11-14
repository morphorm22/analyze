#pragma once

#include "FadTypes.hpp"
#include "ScalarProduct.hpp"
#include "InterpolateFromNodal.hpp"

namespace Plato
{

namespace Parabolic
{

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    TemperatureAverage<EvaluationType, IndicatorFunctionType>::
    TemperatureAverage(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction) {}
    /**************************************************************************/

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    TemperatureAverage<EvaluationType, IndicatorFunctionType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>    & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType> & aStateDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>  & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>   & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>   & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      using TScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ControlScalarType>;

      auto tNumCells   = mSpatialDomain.numCells();
      auto tCubPoints  = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints  = tCubWeights.size();

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

      auto applyWeighting  = mApplyWeighting;
      Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          auto tCubPoint  = tCubPoints(iGpOrdinal);
          auto tCubWeight = tCubWeights(iGpOrdinal);

          auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

          ResultScalarType tCellVolume = Plato::determinant(tJacobian);

          tCellVolume *= tCubWeight;

          auto tBasisValues = ElementType::basisValues(tCubPoint);

          // compute temperature at Gauss points
          //
          TScalarType tState = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aState);

          // apply weighting
          //
          applyWeighting(iCellOrdinal, aControl, tBasisValues, tState);

          Kokkos::atomic_add(&aResult(iCellOrdinal), tState*tCellVolume);
      });
    }
} // namespace Parabolic

} // namespace Plato
