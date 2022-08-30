#pragma once

#include "FadTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "Plato_TopOptFunctors.hpp"

#include <math.h> // need PI

namespace Plato
{

    /**************************************************************************/
    template<typename EvaluationType>
    IntermediateDensityPenalty<EvaluationType>::
    IntermediateDensityPenalty(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams,
              std::string              aFunctionName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aInputParams, aFunctionName),
        mPenaltyAmplitude(1.0)
    /**************************************************************************/
    {
        auto tInputs = aInputParams.sublist("Criteria").sublist(aFunctionName);
        mPenaltyAmplitude = tInputs.get<Plato::Scalar>("Penalty Amplitude", 1.0);
    }

    /**************************************************************************
     * Unit testing constructor
     **************************************************************************/
    template<typename EvaluationType>
    IntermediateDensityPenalty<EvaluationType>::
    IntermediateDensityPenalty(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "IntermediateDensityPenalty"),
        mPenaltyAmplitude(1.0)
    /**************************************************************************/
    {
    }

    /**************************************************************************/
    template<typename EvaluationType>
    void
    IntermediateDensityPenalty<EvaluationType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT<ResultScalarType>       & aResult,
              Plato::Scalar                                  aTimeStep
    ) const 
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      auto tPenaltyAmplitude = mPenaltyAmplitude;

      Plato::Scalar tOne = 1.0;
      Plato::Scalar tTwo = 2.0;
      Plato::Scalar tPi  = M_PI;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      Kokkos::parallel_for("density penalty", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        auto tCubPoint = tCubPoints(iGpOrdinal);

        auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

        auto tVolume = Plato::determinant(tJacobian);

        auto tBasisValues = ElementType::basisValues(tCubPoint);
        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControl);

        ResultScalarType tResult = tVolume * tPenaltyAmplitude / tTwo * (tOne - cos(tTwo * tPi * tCellMass));

        Kokkos::atomic_add(&aResult(iCellOrdinal), tResult);

      });
    }
}
// namespace Plato
