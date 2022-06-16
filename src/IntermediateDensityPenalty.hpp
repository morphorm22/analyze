#pragma once

#include "PlatoStaticsTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "WorksetBase.hpp"
#include "Plato_TopOptFunctors.hpp"
#include <Teuchos_ParameterList.hpp>

#include <math.h> // need PI

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class IntermediateDensityPenalty :
    public EvaluationType::ElementType,
    public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mPenaltyAmplitude;

  public:
    /**************************************************************************/
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
    /**************************************************************************/
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
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT<ResultScalarType>       & aResult,
              Plato::Scalar                                  aTimeStep = 0.0
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
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
};
// class IntermediateDensityPenalty

}
// namespace Plato

#ifdef PLATOANALYZE_1D
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATOANALYZE_2D
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATOANALYZE_3D
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
// TODO extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
