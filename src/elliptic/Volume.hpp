#pragma once

#include "Simplex.hpp"
#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "elliptic/EllipticSimplexFadTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "ExpInstMacros.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename PenaltyFunctionType>
class Volume : public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;
 
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mQuadratureWeight;

    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyWeighting<SpaceDim,1,PenaltyFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    Volume(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList &, 
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mPenaltyFunction(aPenaltyParams),
        mApplyWeighting(mPenaltyFunction)
    /**************************************************************************/
    {
// TODO fix quadrature
      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDimIndex=2; tDimIndex<=SpaceDim; tDimIndex++)
      { 
        mQuadratureWeight /= Plato::Scalar(tDimIndex);
      }
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT<StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;

      auto tQuadratureWeight = mQuadratureWeight;
      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tQuadratureWeight;

        aResult(aCellOrdinal) = tCellVolume;

        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, aResult, aControl);
    
      },"volume");
    }
};
// class Volume

} // namespace Elliptic

} // namespace Plato
