#pragma once

#include "Simplex.hpp"
#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "geometric/GeometricSimplexFadTypes.hpp"
#include "geometric/AbstractScalarFunction.hpp"
#include "UtilsTeuchos.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "ExpInstMacros.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************/
template<typename EvaluationType, typename PenaltyFunctionType>
class Volume : public Plato::Geometric::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::Geometric::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Geometric::AbstractScalarFunction<EvaluationType>::mDataMap;

    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mQuadratureWeight;

    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyWeighting<SpaceDim,1,PenaltyFunctionType> mApplyWeighting;

    bool mCompute;

  public:
    /**************************************************************************/
    Volume(
        const Plato::SpatialDomain   & aSpatialDomain, 
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputs, 
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Geometric::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mPenaltyFunction(aPenaltyParams),
        mApplyWeighting(mPenaltyFunction)
    /**************************************************************************/
    {

      // decide whether we should compute the volume for the current domain
      mCompute = false;
      std::string tCurrentDomainName = aSpatialDomain.getDomainName();

      auto tMyCriteria = aInputs.sublist("Criteria").sublist("Volume");
      std::vector<std::string> tDomains = Plato::teuchos::parse_array<std::string>("Domains", tMyCriteria);

      // see if this matches any of the domains we want to compute volumes of
      for (int i = 0; i < tDomains.size(); i++)
      {
        if (tCurrentDomainName == tDomains[i])
        {
          std::cout<<"tDomains[i] = "<<tDomains[i]<<std::endl;
          mCompute = true;
        }
      }

      // check for a case we don't handle
      if (tMyCriteria.isParameter("Domains") && tDomains.size() == 0)
      {
        THROWERR(std::string("Empty Domains in Volume criteria, either have at least one domain specified or remove from input. No specification indicates we include all domains in volume computation"));
      }

      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDimIndex=2; tDimIndex<=SpaceDim; tDimIndex++)
      { 
        mQuadratureWeight /= Plato::Scalar(tDimIndex);
      }
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult
    ) const override
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;

      auto tQuadratureWeight = mQuadratureWeight;
      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
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

} // namespace Geometric

} // namespace Plato
