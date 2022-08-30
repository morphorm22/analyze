#pragma once

#include "elliptic/StressPNorm_decl.hpp"

#include "FadTypes.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "PlatoMeshExpr.hpp"
#include "GradientMatrix.hpp"

namespace Plato
{

namespace Elliptic
{

    template<typename EvaluationType, typename IndicatorFunctionType>
    StressPNorm<EvaluationType, IndicatorFunctionType>::StressPNorm(
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
      Plato::ElasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());

      auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(params);

      if (params.isType<std::string>("Function"))
        mFuncString = params.get<std::string>("Function");
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    StressPNorm<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        auto tNumCells = mSpatialDomain.numCells();
      
        Plato::ScalarMultiVectorT<ConfigScalarType> tFxnValues("function values", tNumCells*tNumPoints, 1);

        if (mFuncString == "1.0")
        {
            Kokkos::deep_copy(tFxnValues, 1.0);
        }
        else
        {
            Plato::ScalarArray3DT<ConfigScalarType> tPhysicalPoints("physical points", tNumCells, tNumPoints, mNumSpatialDims);
            Plato::mapPoints<ElementType>(aConfig, tPhysicalPoints);

            Plato::getFunctionValues<mNumSpatialDims>(tPhysicalPoints, mFuncString, tFxnValues);
        }

        using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

        Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);

        Plato::ComputeGradientMatrix<ElementType> computeGradient;
        Plato::SmallStrain<ElementType>           computeVoigtStrain;

        Plato::LinearStress<EvaluationType, ElementType> computeVoigtStress(mMaterialModel);

        auto applyWeighting = mApplyWeighting;
        Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrain(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

            computeVoigtStrain(iCellOrdinal, tStrain, aState, tGradient);

            computeVoigtStress(tStress, tStrain);

            tVolume *= tCubWeights(iGpOrdinal);
            tVolume *= tFxnValues(iCellOrdinal*tNumPoints + iGpOrdinal, 0);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            applyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

            for(int i=0; i<ElementType::mNumVoigtTerms; i++)
            {
                Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
            }
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(int i=0; i<ElementType::mNumVoigtTerms; i++)
            {
                tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
            }
        });


        mNorm->evaluate(aResult, tCellStress, aControl, tCellVolume);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    StressPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
        mNorm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    StressPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
        mNorm->postEvaluate(resultValue);
    }

} // namespace Elliptic

} // namespace Plato
