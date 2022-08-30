#pragma once

#include "elliptic/EffectiveEnergy_decl.hpp"

#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "SmallStrain.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "HomogenizedStress.hpp"
#include "ElasticModelFactory.hpp"

namespace Plato
{

namespace Elliptic
{
    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    EffectiveEnergy<EvaluationType, IndicatorFunctionType>::EffectiveEnergy(
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
      auto materialModel = mmfactory.create(aSpatialDomain.getMaterialName());
      mCellStiffness = materialModel->getStiffnessMatrix();

      Teuchos::ParameterList& tParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
      auto tAssumedStrain = tParams.get<Teuchos::Array<Plato::Scalar>>("Assumed Strain");
      assert(tAssumedStrain.size() == mNumVoigtTerms);
      for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
      {
          mAssumedStrain[iVoigt] = tAssumedStrain[iVoigt];
      }

      // parse cell problem forcing
      //
      if(aProblemParams.isSublist("Cell Problem Forcing"))
      {
          mColumnIndex = aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
      }
      else
      {
          ANALYZE_THROWERR("Required input missing: 'Column Index' not given in the 'Cell Problem Forcing' block");
      }

      if( tParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = tParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void 
    EffectiveEnergy<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
        using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

        auto tNumCells = mSpatialDomain.numCells();

        Plato::SmallStrain<ElementType>           voigtStrain;
        Plato::ScalarProduct<mNumVoigtTerms>      scalarProduct;
        Plato::ComputeGradientMatrix<ElementType> computeGradient;
        Plato::HomogenizedStress<ElementType>     homogenizedStress(mCellStiffness, mColumnIndex);

        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);
  
        Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto applyWeighting = mApplyWeighting;
        auto assumedStrain    = mAssumedStrain;
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrain(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

            // compute strain
            //
            voigtStrain(iCellOrdinal, tStrain, aState, tGradient);

            // compute stress
            //
            homogenizedStress(iCellOrdinal, tStress, tStrain);

            // apply weighting
            //
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            applyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
    

            // compute element internal energy (inner product of strain and weighted stress)
            //
            tVolume *= tCubWeights(iGpOrdinal);
            scalarProduct(iCellOrdinal, aResult, tStress, assumedStrain, tVolume);

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

     if( std::count(mPlottable.begin(),mPlottable.end(),"effective stress") ) toMap(mDataMap, tCellStress, "effective stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"cell volume") ) toMap(mDataMap, tCellVolume, "cell volume", mSpatialDomain);

    }

} // namespace Elliptic

} // namespace Plato
