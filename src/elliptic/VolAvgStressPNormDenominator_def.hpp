#pragma once

#include "VolAvgStressPNormDenominator_decl.hpp"

#include "BLAS2.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "ScalarProduct.hpp"
#include "PlatoMeshExpr.hpp"
#include "GradientMatrix.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

namespace Elliptic
{

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    VolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
    VolAvgStressPNormDenominator(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    /**************************************************************************/
    {
      auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(params);

      if (params.isType<std::string>("Function"))
        mSpatialWeightFunction = params.get<std::string>("Function");
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    VolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      auto tSpatialWeights = Plato::computeSpatialWeights<ConfigScalarType, ElementType>(mSpatialDomain, aConfig, mSpatialWeightFunction);

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight", tNumCells);

      Plato::ScalarMultiVectorT<ResultScalarType> tCellWeights("weighted one", tNumCells, mNumVoigtTerms);

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto applyWeighting = mApplyWeighting;
      Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        auto tCubPoint  = tCubPoints(iGpOrdinal);
        auto tCubWeight = tCubWeights(iGpOrdinal);

        // compute cell volume
        //
        auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

        ConfigScalarType tVolume = Plato::determinant(tJacobian) * tCubWeight * tSpatialWeights(iCellOrdinal*tNumPoints + iGpOrdinal, 0);

        Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);

        // apply weighting
        //
        ResultScalarType tWeightedOne(1.0);
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        applyWeighting(iCellOrdinal, aControl, tBasisValues, tWeightedOne);

        Kokkos::atomic_add(&tCellWeights(iCellOrdinal, 0), tWeightedOne);

      });

      mNorm->evaluate(aResult, tCellWeights, aControl, tCellVolume);

    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    VolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    VolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
    postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultValue);
    }
} // namespace Elliptic

} // namespace Plato
