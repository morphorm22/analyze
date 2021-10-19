#ifndef TMSTRESS_P_NORM_HPP
#define TMSTRESS_P_NORM_HPP

#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "TMKinematics.hpp"
#include "TMKineticsFactory.hpp"
#include "TensorPNorm.hpp"
#include "ElasticModelFactory.hpp"
#include "ImplicitFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ThermoelasticMaterial.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "alg/Basis.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "PlatoMeshExpr.hpp"
#include "UtilsOmegaH.hpp"
#include "alg/Cubature.hpp"
#include <Omega_h_expr.hpp>
#include <Omega_h_mesh.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TMStressPNorm : 
  public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    static constexpr int TDofOffset = SpaceDim;

    using PhysicsType = typename Plato::SimplexThermomechanics<SpaceDim>;
    
    using PhysicsType::mNumVoigtTerms;
    using PhysicsType::mNumDofsPerCell;
    using PhysicsType::mNumDofsPerNode;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    Teuchos::RCP<Plato::MaterialModel<SpaceDim>> mMaterialModel;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

    std::string mFuncString = "1.0";

    Omega_h::Reals mFxnValues;

    void computeSpatialWeightingValues(const Plato::SpatialDomain & aSpatialDomain)
    {
      // get refCellQuadraturePoints, quadratureWeights
      //
      Plato::OrdinalType tQuadratureDegree = 1;

      Plato::OrdinalType tNumPoints = Plato::Cubature::getNumCubaturePoints(SpaceDim, tQuadratureDegree);

      Kokkos::View<Plato::Scalar**, Plato::Layout, Plato::MemSpace>
          tRefCellQuadraturePoints("ref quadrature points", tNumPoints, SpaceDim);
      Kokkos::View<Plato::Scalar*, Plato::Layout, Plato::MemSpace> tQuadratureWeights("quadrature weights", tNumPoints);

      Plato::Cubature::getCubature(SpaceDim, tQuadratureDegree, tRefCellQuadraturePoints, tQuadratureWeights);

      // get basis values
      //
      Plato::Basis tBasis(SpaceDim);
      Plato::OrdinalType tNumFields = tBasis.basisCardinality();
      Kokkos::View<Plato::Scalar**, Plato::Layout, Plato::MemSpace>
          tRefCellBasisValues("ref basis values", tNumFields, tNumPoints);
      tBasis.getValues(tRefCellQuadraturePoints, tRefCellBasisValues);

      // map points to physical space
      //
      Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
      Kokkos::View<Plato::Scalar***, Plato::Layout, Plato::MemSpace>
          tQuadraturePoints("quadrature points", tNumCells, tNumPoints, SpaceDim);

      Plato::mapPoints<SpaceDim>(aSpatialDomain, tRefCellQuadraturePoints, tQuadraturePoints);

      // get integrand values at quadrature points
      //
      Plato::getFunctionValues<SpaceDim>(tQuadraturePoints, mFuncString, mFxnValues);
    }

  public:
    /**************************************************************************/
    TMStressPNorm(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
      Plato::ThermoelasticModelFactory<SpaceDim> tFactory(aProblemParams);
      mMaterialModel = tFactory.create(aSpatialDomain.getMaterialName());

      auto tParams = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> tNormFactory;
      mNorm = tNormFactory.create(tParams);

      if (tParams.isType<std::string>("Function"))
        mFuncString = tParams.get<std::string>("Function");
      
      this->computeSpatialWeightingValues(aSpatialDomain);
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<SpaceDim> tComputeGradient;
      Plato::TMKinematics<SpaceDim>           tKinematics;
      //Plato::TMKinetics<SpaceDim>             tKinetics(mMaterialModel);
      Plato::TMKineticsFactory< EvaluationType, Plato::SimplexThermomechanics<EvaluationType::SpatialDim> > tTMKineticsFactory;
      auto pkinetics = tTMKineticsFactory.create(mMaterialModel);
      auto & tKinetics = *pkinetics;

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;

      using GradScalarType = 
        typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
                            StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight", tNumCells);

      Plato::ScalarMultiVectorT<ResultScalarType> tStress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType>   tStrain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tFlux  ("flux",   tNumCells, SpaceDim);
      Plato::ScalarMultiVectorT<GradScalarType>   tTgrad ("tgrad",  tNumCells, SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType> tGradient("gradient", tNumCells, mNumNodesPerCell, SpaceDim);

      Plato::ScalarVectorT<StateScalarType> tTemperature("Gauss point temperature", tNumCells);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tApplyStressWeighting = mApplyStressWeighting;
      auto tFxnValues       = mFxnValues;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight * tFxnValues[aCellOrdinal];

        // compute strain
        //
        tKinematics(aCellOrdinal, tStrain, tTgrad, aState, tGradient);

        // compute stress
        //
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aState, tTemperature);
//        tKinetics(aCellOrdinal, tStress, tFlux, tStrain, tTgrad, tTemperature);
      },"Compute Stress");

      tKinetics(tStress, tFlux, tStrain, tTgrad, tTemperature, aControl);

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
      {
        // apply weighting
        //
        tApplyStressWeighting(aCellOrdinal, tStress, aControl);

      },"Apply Stress Weighting");


      mNorm->evaluate(aResult, tStress, aControl, tCellVolume);

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
PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 3)
#endif

#endif
