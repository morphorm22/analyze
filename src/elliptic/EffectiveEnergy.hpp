#ifndef EFFECTIVE_ELASTIC_ENERGY_HPP
#define EFFECTIVE_ELASTIC_ENERGY_HPP

#include "SimplexMechanics.hpp"
#include "SimplexFadTypes.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "Strain.hpp"
#include "ElasticModelFactory.hpp"
#include "ImplicitFunctors.hpp"
#include "HomogenizedStress.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ExpInstMacros.hpp"
#include "ToMap.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Compute internal effective energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class EffectiveEnergy : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<mSpaceDim>::mNumDofsPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mFunctionName;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim,mNumVoigtTerms,IndicatorFunctionType> mApplyWeighting;

    Plato::Matrix< mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;
    Plato::Array<mNumVoigtTerms> mAssumedStrain;
    Plato::OrdinalType mColumnIndex;
    Plato::Scalar mQuadratureWeight;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    EffectiveEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
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
          // JR TODO: throw
      }

      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType d=2; d<=mSpaceDim; d++)
      { 
          mQuadratureWeight /= Plato::Scalar(d);
      }
    
      if( tParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = tParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
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

      Plato::Strain<mSpaceDim> voigtStrain;
      Plato::ScalarProduct<mNumVoigtTerms> scalarProduct;
      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::HomogenizedStress < mSpaceDim > homogenizedStress(mCellStiffness, mColumnIndex);

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",tNumCells);

      Kokkos::View<StrainScalarType**, Plato::Layout, Plato::MemSpace>
        strain("strain", tNumCells, mNumVoigtTerms);

      Kokkos::View<ConfigScalarType***, Plato::Layout, Plato::MemSpace>
        gradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        stress("stress", tNumCells, mNumVoigtTerms);

      auto quadratureWeight = mQuadratureWeight;
      auto applyWeighting   = mApplyWeighting;
      auto assumedStrain    = mAssumedStrain;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute strain
        //
        voigtStrain(aCellOrdinal, strain, aState, gradient);

        // compute stress
        //
        homogenizedStress(aCellOrdinal, stress, strain);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, stress, aControl);
    
        // compute element internal energy (inner product of strain and weighted stress)
        //
        scalarProduct(aCellOrdinal, aResult, stress, assumedStrain, cellVolume);

      },"energy gradient");

     if( std::count(mPlottable.begin(),mPlottable.end(),"effective stress") ) toMap(mDataMap, stress, "effective stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"cell volume") ) toMap(mDataMap, cellVolume, "cell volume", mSpatialDomain);

    }
};
// class EffectiveEnergy

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Elliptic::EffectiveEnergy, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::EffectiveEnergy, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::EffectiveEnergy, Plato::SimplexMechanics, 3)
#endif

#endif
