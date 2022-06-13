#ifndef INTERNAL_THERMOELASTIC_ENERGY_HPP
#define INTERNAL_THERMOELASTIC_ENERGY_HPP

#include "FadTypes.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "ImplicitFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "ThermoelasticMaterial.hpp"
#include "ToMap.hpp"
#include "ExpInstMacros.hpp"

namespace Plato
{

namespace Elliptic
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
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    static constexpr int TDofOffset = mNumSpatialDims;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyFluxWeighting;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

  public:
    /**************************************************************************/
    InternalThermoelasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyFluxWeighting   (mIndicatorFunction)
    /**************************************************************************/
    {
        Teuchos::ParameterList tProblemParams(aProblemParams);

        auto tMaterialName = aSpatialDomain.getMaterialName();

        if( aProblemParams.isSublist("Material Models") == false )
        {
            ANALYZE_THROWERR("Required input list ('Material Models') is missing.");
        }

        if( aProblemParams.sublist("Material Models").isSublist(tMaterialName) == false )
        {
            std::stringstream ss;
            ss << "Specified material model ('" << tMaterialName << "') is not defined";
            ANALYZE_THROWERR(ss.str());
        }

        auto& tParams = aProblemParams.sublist(aFunctionName);
        if( tParams.get<bool>("Include Thermal Strain", true) == false )
        {
           auto tMaterialParams = tProblemParams.sublist("Material Models").sublist(tMaterialName);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a11",0.0);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a22",0.0);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a33",0.0);
        }

        Plato::ThermoelasticModelFactory<mNumSpatialDims> mmfactory(tProblemParams);
        mMaterialModel = mmfactory.create(tMaterialName);
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
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::TMKinematics<ElementType>          kinematics;
      Plato::TMKinetics<ElementType>            kinetics(mMaterialModel);

      Plato::ScalarProduct<mNumVoigtTerms>      mechanicalScalarProduct;
      Plato::ScalarProduct<mNumSpatialDims>     thermalScalarProduct;

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

          // compute element internal energy
          //
          mechanicalScalarProduct(iCellOrdinal, aResult, tStress, tStrain, tVolume);
          thermalScalarProduct   (iCellOrdinal, aResult, tFlux,   tTGrad,  tVolume);

        });
    }
};

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 3)
#endif

#endif
