#pragma once

#include "elliptic/FluxPNorm_decl.hpp"

#include "FadTypes.hpp"
#include "ScalarGrad.hpp"
#include "VectorPNorm.hpp"
#include "ThermalFlux.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"


namespace Plato
{

namespace Elliptic
{

    template<typename EvaluationType, typename IndicatorFunctionType>
    FluxPNorm<EvaluationType, IndicatorFunctionType>::FluxPNorm(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
              std::string              aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction(aPenaltyParams),
        mApplyWeighting(mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ThermalConductionModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(mSpatialDomain.getMaterialName());

      auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      mExponent = params.get<Plato::Scalar>("Exponent");
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void FluxPNorm<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
      Plato::ScalarGrad<ElementType>            tScalarGrad;
      Plato::ThermalFlux<ElementType>           thermalFlux(mMaterialModel);
      Plato::VectorPNorm<mNumSpatialDims>       tVectorPNorm;

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& tApplyWeighting = mApplyWeighting;
      auto tExponent        = mExponent;
      Kokkos::parallel_for("thermal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
          tScalarGrad(iCellOrdinal, tGrad, aState, tGradient);

          // compute flux
          //
          StateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aState);
          thermalFlux(tFlux, tGrad, tTemperature);

          // apply weighting
          //
          tApplyWeighting(iCellOrdinal, aControl, tBasisValues, tFlux);
    
          // compute vector p-norm of flux
          //
          tVectorPNorm(iCellOrdinal, aResult, tFlux, tExponent, tVolume);

      });
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    FluxPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      auto scale = pow(resultScalar,(1.0-mExponent)/mExponent)/mExponent;
      auto numEntries = resultVector.size();
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numEntries), LAMBDA_EXPRESSION(int entryOrdinal)
      {
        resultVector(entryOrdinal) *= scale;
      },"scale vector");
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    FluxPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      resultValue = pow(resultValue, 1.0/mExponent);
    }
} // namespace Elliptic

} // namespace Plato
