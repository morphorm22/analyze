#pragma once

#include "elliptic/InternalThermalEnergy_decl.hpp"

#include "FadTypes.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Constructor
     * \param aSpatialDomain Plato Analyze spatial domain
     * \param aProblemParams input database for overall problem
     * \param aPenaltyParams input database for penalty function
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    InternalThermalEnergy<EvaluationType, IndicatorFunctionType>::InternalThermalEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    {
        Plato::ThermalConductionModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());
    }

    /******************************************************************************//**
     * \brief Evaluate internal elastic energy function
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void InternalThermalEnergy<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    {
      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
      Plato::ScalarGrad<ElementType>            tComputeScalarGrad;
      Plato::ScalarProduct<mNumSpatialDims>     tComputeScalarProduct;
      Plato::ThermalFlux<ElementType>           tComputeThermalFlux(mMaterialModel);

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for("thermal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumSpatialDims, GradScalarType> tGrad(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tFlux(0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);
          auto tBasisValues = ElementType::basisValues(tCubPoint);

          tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          tVolume *= tCubWeights(iGpOrdinal);

          // compute temperature gradient
          //
          tComputeScalarGrad(iCellOrdinal, tGrad, aState, tGradient);

          // compute flux
          //
          StateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aState);
          tComputeThermalFlux(tFlux, tGrad, tTemperature);

          // apply weighting
          //
          tApplyWeighting(iCellOrdinal, aControl, tBasisValues, tFlux);
    
          // compute element internal energy (inner product of tgrad and weighted tflux)
          //
          tComputeScalarProduct(iCellOrdinal, aResult, tFlux, tGrad, tVolume, -1.0);

      });
    }
} // namespace Elliptic

} // namespace Plato
