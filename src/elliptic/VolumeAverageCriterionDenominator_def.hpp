#pragma once

#include "VolumeAverageCriterionDenominator_decl.hpp"

#include "PlatoMeshExpr.hpp"

namespace Plato
{

namespace Elliptic
{

    /**************************************************************************/
    template<typename EvaluationType>
    VolumeAverageCriterionDenominator<EvaluationType>::
    VolumeAverageCriterionDenominator(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              std::string            & aFunctionName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mSpatialWeightFunction("1.0")
    /**************************************************************************/
    {}

    /******************************************************************************//**
     * \brief Set spatial weight function
     * \param [in] aInput math expression
    **********************************************************************************/
    template<typename EvaluationType>
    void
    VolumeAverageCriterionDenominator<EvaluationType>::
    setSpatialWeightFunction(std::string aWeightFunctionString)
    {
        mSpatialWeightFunction = aWeightFunctionString;
    }

    /**************************************************************************/
    template<typename EvaluationType>
    void
    VolumeAverageCriterionDenominator<EvaluationType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
        auto tSpatialWeights  = Plato::computeSpatialWeights<ConfigScalarType, ElementType>(mSpatialDomain, aConfig, mSpatialWeightFunction);

        auto tNumCells = mSpatialDomain.numCells();
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint  = tCubPoints(iGpOrdinal);
            auto tCubWeight = tCubWeights(iGpOrdinal);

            auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

            ResultScalarType tCellVolume = Plato::determinant(tJacobian);

            tCellVolume *= tCubWeight;

            Kokkos::atomic_add(&aResult(iCellOrdinal), tCellVolume*tSpatialWeights(iCellOrdinal * tNumPoints + iGpOrdinal, 0));
        });
    }
} // namespace Elliptic

} // namespace Plato

