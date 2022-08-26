#pragma once

#include "FadTypes.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"

namespace Plato
{

namespace Parabolic
{
    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    InternalThermalEnergy<EvaluationType, IndicatorFunctionType>::
    InternalThermalEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ThermalConductionModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
      mThermalConductivityMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    InternalThermalEnergy<EvaluationType, IndicatorFunctionType>::
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
      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::ScalarGrad<ElementType>            scalarGrad;
      Plato::ScalarProduct<mNumSpatialDims>     scalarProduct;
      Plato::ThermalFlux<ElementType>           thermalFlux(mThermalConductivityMaterialModel);

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto applyWeighting = mApplyWeighting;
      Kokkos::parallel_for("thermal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumSpatialDims, GradScalarType> tGrad(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tFlux(0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);
          auto tBasisValues = ElementType::basisValues(tCubPoint);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);
          tVolume *= tCubWeights(iGpOrdinal);

          // compute temperature gradient
          //
          scalarGrad(iCellOrdinal, tGrad, aState, tGradient);

          // compute flux
          //
          StateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aState);
          thermalFlux(tFlux, tGrad, tTemperature);

          // apply weighting
          //
          applyWeighting(iCellOrdinal, aControl, tBasisValues, tFlux);
    
          // compute element internal energy (inner product of tgrad and weighted tflux)
          //
          scalarProduct(iCellOrdinal, aResult, tFlux, tGrad, tVolume, -1.0);

      });
    }
} // namespace Parabolic

} // namespace Plato
