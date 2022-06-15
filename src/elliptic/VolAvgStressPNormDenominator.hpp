#pragma once

#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "TensorPNorm.hpp"
#include "ElasticModelFactory.hpp"
#include "ImplicitFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ExpInstMacros.hpp"

#include "alg/Basis.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "PlatoMeshExpr.hpp"
#include "alg/Cubature.hpp"
#include "BLAS2.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class VolAvgStressPNormDenominator : 
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, /*number of terms=*/1, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

    std::string mSpatialWeightFunction = "1.0";

  public:
    /**************************************************************************/
    VolAvgStressPNormDenominator(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
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
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    /**************************************************************************/
    {
      auto tSpatialWeights = Plato::computeSpatialWeights<ConfigScalarType, ElementType>(mSpatialDomain, aConfig, mSpatialWeightFunction);

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight", tNumCells);

      Plato::ScalarMultiVectorT<ResultScalarType> tCellWeights("weighted one", tNumCells, mNumVoigtTerms);

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto applyWeighting = mApplyWeighting;
      Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
    void
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    void
    postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultValue);
    }
};
// class VolAvgStressPNormDenominator

} // namespace Elliptic

} // namespace Plato


#ifdef PLATOANALYZE_2D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::VolAvgStressPNormDenominator, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::VolAvgStressPNormDenominator, Plato::SimplexMechanics, 3)
#endif
