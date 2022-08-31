#pragma once

#include "Volume_decl.hpp"

namespace Plato
{

namespace Elliptic
{

    /**************************************************************************/
    template<typename EvaluationType, typename PenaltyFunctionType>
    Volume<EvaluationType, PenaltyFunctionType>::
    Volume(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mPenaltyFunction (aPenaltyParams),
        mApplyWeighting  (mPenaltyFunction)
    /**************************************************************************/
    {}

    /**************************************************************************/
    template<typename EvaluationType, typename PenaltyFunctionType>
    void
    Volume<EvaluationType, PenaltyFunctionType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT<StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
        auto tNumCells = mSpatialDomain.numCells();
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        auto tApplyWeighting  = mApplyWeighting;

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
} // namespace Elliptic

} // namespace Plato
