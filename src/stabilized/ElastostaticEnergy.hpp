#pragma once

#include "GradientMatrix.hpp"
#include "stabilized/MechanicsElement.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ScalarProduct.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "Kinematics.hpp"
#include "Kinetics.hpp"
#include "ImplicitFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ElasticModelFactory.hpp"
#include "ToMap.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************//**
 * \brief Compute internal elastic energy criterion for stabilized form.
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticEnergy : 
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNMechDims  = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNPressDims = 1;

    static constexpr Plato::OrdinalType mMDofOffset = 0;
    static constexpr Plato::OrdinalType mPressDofOffset = mNumSpatialDims;
    
    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
//    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyTensorWeighting;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

  public:
    /**************************************************************************/
    ElastostaticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap&          aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyTensorWeighting (mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
    }

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarVectorT      <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix  <ElementType> tComputeGradient;
      Plato::Stabilized::Kinematics <ElementType> tKinematics;
      Plato::Stabilized::Kinetics   <ElementType> tKinetics(mMaterialModel);

      Plato::InterpolateFromNodal   <ElementType, mNumDofsPerNode, mPressDofOffset> tInterpolatePressureFromNodal;

      Plato::ScalarProduct<mNumVoigtTerms> tDeviatorScalarProduct;
      
      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& tApplyTensorWeighting = mApplyTensorWeighting;

      Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        ConfigScalarType tVolume(0.0);

        Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;


        // compute gradient operator and cell volume
        //
        auto tCubPoint = tCubPoints(iGpOrdinal);
        tComputeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
        tVolume *= tCubWeights(iGpOrdinal);

        // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
        //
        Plato::Array<mNumVoigtTerms, GradScalarType> tDGrad(0.0);
        Plato::Array<mNumSpatialDims, GradScalarType> tPGrad(0.0);
        tKinematics(iCellOrdinal, tDGrad, tPGrad, aStateWS, tGradient);

        // interpolate projected pressure to gauss point
        //
        ResultScalarType tPressure(0.0);
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        tInterpolatePressureFromNodal(iCellOrdinal, tBasisValues, aStateWS, tPressure);

        // compute the constitutive response
        //
        ResultScalarType tVolStrain(0.0);
        Plato::Array<mNumSpatialDims, ResultScalarType> tCellStab(0.0);
//        Plato::Array<mNumSpatialDims, NodeStateScalarType> tProjectedPGrad(0.0);
        Plato::Array<mNumSpatialDims, ResultScalarType> tProjectedPGrad(0.0);
        Plato::Array<mNumVoigtTerms, ResultScalarType> tDevStress(0.0);
        tKinetics(tVolume, tProjectedPGrad, tDGrad, tPGrad,
                  tPressure, tDevStress, tVolStrain, tCellStab);

        Plato::Array<mNumVoigtTerms, ResultScalarType> tTotStress(0.0);
        for( int i=0; i<mNumSpatialDims; i++)
        {
            tTotStress(i) = tDevStress(i) + tPressure;
        }

        // apply weighting
        //
        tApplyTensorWeighting (iCellOrdinal, aControlWS, tBasisValues, tTotStress);

        // compute element internal energy (inner product of strain and weighted stress)
        //
        tDeviatorScalarProduct(iCellOrdinal, aResultWS, tTotStress, tDGrad, tVolume);
      });
    }
};
// class InternalElasticEnergy

} // namespace Stabilized
} // namespace Plato

#ifdef PLATOANALYZE_1D
//TODO PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
//TODO PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
//TODO PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 3)
#endif
