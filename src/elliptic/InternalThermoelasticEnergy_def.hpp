#pragma once

#include "elliptic/InternalThermoelasticEnergy_decl.hpp"

#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "TMKinetics.hpp"
#include "TMKinematics.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"

namespace Plato
{

namespace Elliptic
{

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    InternalThermoelasticEnergy<EvaluationType, IndicatorFunctionType>::InternalThermoelasticEnergy(
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
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    InternalThermoelasticEnergy<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
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
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          StateScalarType tTemperature = interpolateFromNodal(iCellOrdinal, tBasisValues, aState);
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
} // namespace Elliptic

} // namespace Plato
