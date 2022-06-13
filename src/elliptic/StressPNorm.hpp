#ifndef STRESS_P_NORM_HPP
#define STRESS_P_NORM_HPP

#include "elliptic/AbstractScalarFunction.hpp"

#include "SmallStrain.hpp"
#include "FadTypes.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "LinearStress.hpp"
#include "TensorPNorm.hpp"
#include "ElasticModelFactory.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "PlatoMeshExpr.hpp"

#include "ElasticModelFactory.hpp"
#include "ExpInstMacros.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class StressPNorm : 
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

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

    std::string mFuncString = "1.0";

    Plato::ScalarMultiVector mFxnValues;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;


  public:
    /**************************************************************************/
    StressPNorm(
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
      Plato::ElasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());

      auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(params);

      if (params.isType<std::string>("Function"))
        mFuncString = params.get<std::string>("Function");
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
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal)
        {
            for(int i=0; i<ElementType::mNumVoigtTerms; i++)
            {
                tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
            }
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
// class StressPNorm

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
//PLATO_EXPL_DEC(Plato::Elliptic::StressPNorm, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
//PLATO_EXPL_DEC(Plato::Elliptic::StressPNorm, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
//PLATO_EXPL_DEC(Plato::Elliptic::StressPNorm, Plato::SimplexMechanics, 3)
#endif

#endif
