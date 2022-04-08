#pragma once

#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"
#include "geometric/EvaluationTypes.hpp"
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
class Volume :
    public EvaluationType::ElementType,
    public Plato::Geometric::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;

    using Plato::Geometric::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Geometric::AbstractScalarFunction<EvaluationType>::mDataMap;

    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyWeighting<mNumNodesPerCell,1,PenaltyFunctionType> mApplyWeighting;

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


      auto tMyCriteria = aInputs.sublist("Criteria").sublist(aFunctionName);
      std::vector<std::string> tDomains = Plato::teuchos::parse_array<std::string>("Domains", tMyCriteria);

      // see if this matches any of the domains we want to compute volumes of
      for (int i = 0; i < tDomains.size(); i++)
      {
        if (tCurrentDomainName == tDomains[i])
        {
          mCompute = true;
        }
      }

      // if not specified compute all
      if(!tMyCriteria.isParameter("Domains"))
      {
          WARNING(std::string("'Domains' parameter is not defined in Volume criterion. All domains will be included in the volume computation."));
          mCompute = true;
      }

      // check for a case we don't handle
      if (tMyCriteria.isParameter("Domains") && tDomains.size() == 0)
      {
          WARNING(std::string("Empty Domains array in Volume criterion. All domains will be included in the volume computation."));
          mCompute = true;
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
        if (mCompute)
        {
            auto tNumCells   = mSpatialDomain.numCells();
            auto tCubPoints  = ElementType::getCubPoints();
            auto tCubWeights = ElementType::getCubWeights();
            auto tNumPoints  = tCubWeights.size();

            auto& tApplyWeighting = mApplyWeighting;

            Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
            LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
            {
                auto tCubPoint  = tCubPoints(iGpOrdinal);
                auto tCubWeight = tCubWeights(iGpOrdinal);

                auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

                ResultScalarType tCellVolume = Plato::determinant(tJacobian);

                tCellVolume *= tCubWeight;

                auto tBasisValues = ElementType::basisValues(tCubPoint);
                tApplyWeighting(iCellOrdinal, aControl, tBasisValues, tCellVolume);

                Kokkos::atomic_add(&aResult(iCellOrdinal), tCellVolume);

            });
        }
    }
};
// class Volume

} // namespace Geometric

} // namespace Plato
