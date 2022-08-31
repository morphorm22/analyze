#pragma once

#include <algorithm>

#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"
#include "geometric/EvaluationTypes.hpp"
#include "geometric/AbstractScalarFunction.hpp"
#include "ExpInstMacros.hpp"

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
        const std::string            & aFunctionName
    ) :
        Plato::Geometric::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aInputs, aFunctionName),
        mPenaltyFunction(aPenaltyParams),
        mApplyWeighting(mPenaltyFunction) {}
    /**************************************************************************/

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult
    ) const override
    /**************************************************************************/
    {
        auto tNumCells   = mSpatialDomain.numCells();
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        auto& tApplyWeighting = mApplyWeighting;

        Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
};
// class Volume

} // namespace Geometric

} // namespace Plato
