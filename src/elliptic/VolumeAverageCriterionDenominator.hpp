#pragma once

#include "ImplicitFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ExpInstMacros.hpp"
#include "BLAS1.hpp"
#include "alg/Basis.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType>
class VolumeAverageCriterionDenominator : 
    public EvaluationType::ElementType,
    public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
    
    std::string mSpatialWeightFunction;

  public:
    /**************************************************************************/
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
    void setSpatialWeightFunction(std::string aWeightFunctionString) override
    {
        mSpatialWeightFunction = aWeightFunctionString;
    }

    /******************************************************************************//**
     * \brief compute spatial weights
     * \param [in] aInput node location data
    **********************************************************************************/
    Plato::ScalarMultiVectorT<ConfigScalarType>
    computeSpatialWeights(
        const Plato::ScalarArray3DT<ConfigScalarType> & aConfig
    ) const
    {
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        auto tNumCells = mSpatialDomain.numCells();

        Plato::ScalarArray3DT<ConfigScalarType> tPhysicalPoints("physical points", tNumCells, tNumPoints, mNumSpatialDims);
        Plato::mapPoints<ElementType>(aConfig, tPhysicalPoints);

        Plato::ScalarMultiVectorT<ConfigScalarType> tFxnValues("function values", tNumCells*tNumPoints, 1);
        Plato::getFunctionValues<mNumSpatialDims>(tPhysicalPoints, mSpatialWeightFunction, tFxnValues);

        return tFxnValues;
    }

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    /**************************************************************************/
    {
        auto tSpatialWeights  = computeSpatialWeights(aConfig);

        auto tNumCells = mSpatialDomain.numCells();
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint  = tCubPoints(iGpOrdinal);
            auto tCubWeight = tCubWeights(iGpOrdinal);

            auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

            ResultScalarType tCellVolume = Plato::determinant(tJacobian);

            tCellVolume *= tCubWeight;

            Kokkos::atomic_add(&aResult(iCellOrdinal), tCellVolume*tSpatialWeights(iCellOrdinal * tNumPoints + iGpOrdinal, 0));
        });
    }
};
// class VolumeAverageCriterionDenominator

} // namespace Elliptic

} // namespace Plato

