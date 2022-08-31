#pragma once

#include "elliptic/EMStressPNorm_decl.hpp"

#include "FadTypes.hpp"
#include "EMKinetics.hpp"
#include "EMKinematics.hpp"
#include "PlatoMeshExpr.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"

namespace Plato
{

namespace Elliptic
{

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    EMStressPNorm<EvaluationType, IndicatorFunctionType>::EMStressPNorm(
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
      Plato::ElectroelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());

      auto tParams = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(tParams);

      if (tParams.isType<std::string>("Function"))
        mFuncString = tParams.get<std::string>("Function");

    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    EMStressPNorm<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

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

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
      Plato::EMKinematics<ElementType>          tKinematics;
      Plato::EMKinetics<ElementType>            tKinetics(mMaterialModel);

      Plato::ScalarVectorT<ConfigScalarType>  tCellVolume("cell weight", tNumCells);

      Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);

      auto tApplyWeighting = mApplyWeighting;
      Kokkos::parallel_for("compute internal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<mNumVoigtTerms,  GradScalarType>   tStrain(0.0);
          Plato::Array<mNumSpatialDims, GradScalarType>   tEField(0.0);
          Plato::Array<mNumVoigtTerms,  ResultScalarType> tStress(0.0);
          Plato::Array<mNumSpatialDims, ResultScalarType> tEDisp (0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          tVolume *= tCubWeights(iGpOrdinal);
          tVolume *= tFxnValues(iCellOrdinal*tNumPoints + iGpOrdinal, 0);

          // compute strain and electric field
          //
          tKinematics(iCellOrdinal, tStrain, tEField, aState, tGradient);

          // compute stress and electric displacement
          //
          tKinetics(tStress, tEDisp, tStrain, tEField);

          // apply weighting
          //
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          tApplyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
          }

          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
      });

      Kokkos::parallel_for("compute cell stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumVoigtTerms}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iVoigtOrdinal)
      {
          tCellStress(iCellOrdinal, iVoigtOrdinal) /= tCellVolume(iCellOrdinal);
      });

      mNorm->evaluate(aResult, tCellStress, aControl, tCellVolume);

    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    EMStressPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar
    )
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    EMStressPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate(
      Plato::Scalar& resultValue
    )
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultValue);
    }
} // namespace Elliptic

} // namespace Plato
