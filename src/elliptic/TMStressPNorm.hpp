#ifndef TMSTRESS_P_NORM_HPP
#define TMSTRESS_P_NORM_HPP

#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "TensorPNorm.hpp"
#include "ElasticModelFactory.hpp"
#include "ImplicitFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "ThermoelasticMaterial.hpp"

#include "alg/Basis.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "PlatoMeshExpr.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TMStressPNorm : 
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    static constexpr int TDofOffset = mNumSpatialDims;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms, EvaluationType>> mNorm;

    std::string mFuncString = "1.0";

  public:
    /**************************************************************************/
    TMStressPNorm(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ThermoelasticModelFactory<mNumSpatialDims> tFactory(aProblemParams);
      mMaterialModel = tFactory.create(aSpatialDomain.getMaterialName());

      auto tParams = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> tNormFactory;
      mNorm = tNormFactory.create(tParams);

      if (tParams.isType<std::string>("Function"))
        mFuncString = tParams.get<std::string>("Function");
    }

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
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
      Plato::TMKinematics<ElementType>          tKinematics;
      Plato::TMKinetics<ElementType>            tKinetics(mMaterialModel);

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;

      Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);

      auto tApplyStressWeighting = mApplyStressWeighting;
      Kokkos::parallel_for("compute internal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<mNumVoigtTerms,  GradScalarType>   tStrain(0.0);
          Plato::Array<mNumSpatialDims, GradScalarType>   tTGrad (0.0);
          Plato::Array<mNumVoigtTerms,  ResultScalarType> tStress(0.0);
          Plato::Array<mNumSpatialDims, ResultScalarType> tFlux  (0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          tVolume *= tCubWeights(iGpOrdinal);
          tVolume *= tFxnValues(iCellOrdinal*tNumPoints + iGpOrdinal, 0);

          // compute strain and electric field
          //
          tKinematics(iCellOrdinal, tStrain, tTGrad, aState, tGradient);

          // compute stress and electric displacement
          //
          StateScalarType tTemperature(0.0);
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          tInterpolateFromNodal(iCellOrdinal, tBasisValues, aState, tTemperature);
          tKinetics(tStress, tFlux, tStrain, tTGrad, tTemperature);

          // apply weighting
          //
          tApplyStressWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
          }

          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
      });

      Kokkos::parallel_for("compute cell stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumVoigtTerms}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iVoigtOrdinal)
      {
          tCellStress(iCellOrdinal, iVoigtOrdinal) /= tCellVolume(iCellOrdinal);
      });

      mNorm->evaluate(aResult, tCellStress, aControl, tCellVolume);

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
// class TMStressPNorm

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 3)
#endif

#endif
