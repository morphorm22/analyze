#pragma once

#include "parabolic/AbstractScalarFunction.hpp"

#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "ThermoelasticMaterial.hpp"
#include "ToMap.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************//**
 * \brief Compute internal thermo-elastic energy criterion, given by
 *                  /f$ f(z) = u^{T}K_u(z)u + T^{T}K_t(z)T /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermoelasticEnergy :
  public EvaluationType::ElementType,
  public Plato::Parabolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    static constexpr int TDofOffset = mNumSpatialDims;

    using FunctionBaseType = Plato::Parabolic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType    = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using ControlScalarType  = typename EvaluationType::ControlScalarType;
    using ConfigScalarType   = typename EvaluationType::ConfigScalarType;
    using ResultScalarType   = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;
    ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyFluxWeighting;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    InternalThermoelasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyFluxWeighting   (mIndicatorFunction)
    /**************************************************************************/
    {
        Plato::ThermoelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
        mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());

        if( aProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = aProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>    & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType> & aStateDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>  & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>   & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>   & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      TMKinematics<ElementType>                 kinematics;
      TMKinetics<ElementType>                   kinetics(mMaterialModel);

      ScalarProduct<mNumVoigtTerms>  mechanicalScalarProduct;
      ScalarProduct<mNumSpatialDims> thermalScalarProduct;

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyFluxWeighting   = mApplyFluxWeighting;
      Kokkos::parallel_for("compute internal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<mNumVoigtTerms,  GradScalarType>   tStrain(0.0);
          Plato::Array<mNumSpatialDims, GradScalarType>   tTGrad (0.0);
          Plato::Array<mNumVoigtTerms,  ResultScalarType> tStress(0.0);
          Plato::Array<mNumSpatialDims, ResultScalarType> tFlux  (0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);
          tVolume *= tCubWeights(iGpOrdinal);

          // compute strain and temperature gradient
          //
          kinematics(iCellOrdinal, tStrain, tTGrad, aState, tGradient);

          // compute stress and thermal flux
          //
          StateScalarType tTemperature(0.0);
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          interpolateFromNodal(iCellOrdinal, tBasisValues, aState, tTemperature);
          kinetics(tStress, tFlux, tStrain, tTGrad, tTemperature);

          // apply weighting
          //
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
          applyFluxWeighting  (iCellOrdinal, aControl, tBasisValues, tFlux);

          // compute element internal energy (inner product of strain and weighted stress)
          //
          mechanicalScalarProduct(iCellOrdinal, aResult, tStress, tStrain, tVolume);
          thermalScalarProduct   (iCellOrdinal, aResult, tFlux,   tTGrad,  tVolume);

      });
    }
};
// class InternalThermoelasticEnergy

} // namespace Parabolic

} // namespace Plato
