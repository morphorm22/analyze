#pragma once

#include "BLAS2.hpp"
#include "PlatoTypes.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "GeneralStressDivergence.hpp"
#include "PressureDivergence.hpp"
#include "ProjectToNode.hpp"
#include "GradientMatrix.hpp"
#include "GeneralFluxDivergence.hpp"
#include "stabilized/AbstractVectorFunction.hpp"
#include "ApplyWeighting.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearThermoelasticMaterial.hpp"
#include "NaturalBCs.hpp"
#include "BodyLoads.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermoelastostaticResidual :
        public EvaluationType::ElementType,
        public Plato::Stabilized::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumSpatialDims;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;

    static constexpr int NMechDims  = mNumSpatialDims;
    static constexpr int NPressDims = 1;
    static constexpr int NThrmDims  = 1;

    static constexpr int MDofOffset = 0;
    static constexpr int PDofOffset = mNumSpatialDims;
    static constexpr int TDofOffset = mNumSpatialDims+1;

    using FunctionBaseType = Plato::Stabilized::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyTensorWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyVectorWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, /*num_dofs=*/1,  IndicatorFunctionType> mApplyScalarWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;

    std::shared_ptr<Plato::NaturalBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>> mBoundaryFluxes;

    Teuchos::RCP<Plato::LinearThermoelasticMaterial<mNumSpatialDims>> mMaterialModel;

public:
    /**************************************************************************/
    ThermoelastostaticResidual(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap& aDataMap,
              Teuchos::ParameterList& aProblemParams,
              Teuchos::ParameterList& aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyTensorWeighting (mIndicatorFunction),
        mApplyVectorWeighting (mIndicatorFunction),
        mApplyScalarWeighting (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr),
        mBoundaryFluxes       (nullptr)
    /**************************************************************************/
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
        if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
        mDofNames.push_back("pressure");
        mDofNames.push_back("temperature");

        // create material model and get stiffness
        //
        Plato::LinearThermoelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
        mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
  

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse mechanical boundary Conditions
        // 
        if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>>
                                (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
  
        // parse thermal boundary Conditions
        // 
        if(aProblemParams.isSublist("Thermal Natural Boundary Conditions"))
        {
            mBoundaryFluxes = std::make_shared<Plato::NaturalBCs<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>>
                                 (aProblemParams.sublist("Thermal Natural Boundary Conditions"));
        }
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    {
      return aSolutions;
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT<StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT<NodeStateScalarType> & aPGradWS,
        const Plato::ScalarMultiVectorT<ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT<ConfigScalarType>        & aConfigWS,
              Plato::ScalarMultiVectorT<ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix    <ElementType> computeGradient;
      Plato::Stabilized::TMKinematics <ElementType> kinematics;
      Plato::Stabilized::TMKinetics   <ElementType> kinetics(mMaterialModel);

      Plato::InterpolateFromNodal <ElementType, mNumSpatialDims, 0, mNumSpatialDims> interpolatePGradFromNodal;
      Plato::InterpolateFromNodal <ElementType, mNumDofsPerNode, PDofOffset> interpolatePressureFromNodal;
      Plato::InterpolateFromNodal <ElementType, mNumDofsPerNode, TDofOffset> interpolateTemperatureFromNodal;
      
      Plato::GeneralFluxDivergence   <ElementType, mNumDofsPerNode, TDofOffset> fluxDivergence;
      Plato::GeneralFluxDivergence   <ElementType, mNumDofsPerNode, PDofOffset> stabDivergence;
      Plato::GeneralStressDivergence <ElementType, mNumDofsPerNode, MDofOffset> stressDivergence;
      Plato::ProjectToNode           <ElementType, mNumDofsPerNode, PDofOffset> projectVolumeStrain;

      Plato::PressureDivergence <ElementType, mNumDofsPerNode> pressureDivergence;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyTensorWeighting = mApplyTensorWeighting;
      auto& applyVectorWeighting = mApplyVectorWeighting;
      auto& applyScalarWeighting = mApplyScalarWeighting;

      Kokkos::parallel_for("compute residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        ConfigScalarType tVolume(0.0);

        Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

        // compute gradient operator and cell volume
        //
        auto tCubPoint = tCubPoints(iGpOrdinal);
        computeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
        tVolume *= tCubWeights(iGpOrdinal);

        // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
        //
        Plato::Array<mNumVoigtTerms, GradScalarType> tDGrad(0.0);
        Plato::Array<mNumSpatialDims, GradScalarType> tPGrad(0.0);
        Plato::Array<mNumSpatialDims, GradScalarType> tTGrad(0.0);
        kinematics(iCellOrdinal, tDGrad, tPGrad, tTGrad, aStateWS, tGradient);

        // interpolate projected PGrad, pressure, and temperature to gauss point
        //
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        Plato::Array<mNumSpatialDims, NodeStateScalarType> tProjectedPGrad(0.0);
        interpolatePGradFromNodal(iCellOrdinal, tBasisValues, aPGradWS, tProjectedPGrad);

        ResultScalarType tPressure;
        interpolatePressureFromNodal(iCellOrdinal, tBasisValues, aStateWS, tPressure);

        ResultScalarType tTemperature;
        interpolateTemperatureFromNodal(iCellOrdinal, tBasisValues, aStateWS, tTemperature);

        // compute the constitutive response
        //
        ResultScalarType tVolStrain(0.0);
        Plato::Array<mNumSpatialDims, ResultScalarType> tCellStab(0.0);
        Plato::Array<mNumSpatialDims, ResultScalarType> tTFlux(0.0);
        Plato::Array<mNumVoigtTerms, ResultScalarType> tDevStress(0.0);
        kinetics(tVolume, tProjectedPGrad, tDGrad, tPGrad, tTGrad, tTemperature,
                 tPressure, tDevStress, tVolStrain, tTFlux, tCellStab);

        // apply weighting
        //
        applyTensorWeighting (iCellOrdinal, aControlWS, tBasisValues, tDevStress);
        applyVectorWeighting (iCellOrdinal, aControlWS, tBasisValues, tCellStab);
        applyVectorWeighting (iCellOrdinal, aControlWS, tBasisValues, tTFlux);
        applyScalarWeighting (iCellOrdinal, aControlWS, tBasisValues, tPressure);
        applyScalarWeighting (iCellOrdinal, aControlWS, tBasisValues, tVolStrain);
    
        // compute divergence
        //
        stressDivergence    (iCellOrdinal, aResultWS,  tDevStress, tGradient, tVolume);
        pressureDivergence  (iCellOrdinal, aResultWS,  tPressure,  tGradient, tVolume);
        stabDivergence      (iCellOrdinal, aResultWS,  tCellStab,  tGradient, tVolume, -1.0);
        fluxDivergence      (iCellOrdinal, aResultWS,  tTFlux,     tGradient, tVolume);

        projectVolumeStrain (iCellOrdinal, tVolume, tBasisValues, tVolStrain, aResultWS);

      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0 );
      }
    }
    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                            & aSpatialModel,
        const Plato::ScalarMultiVectorT<StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT<NodeStateScalarType> & aPGradWS,
        const Plato::ScalarMultiVectorT<ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT<ConfigScalarType>        & aConfigWS,
              Plato::ScalarMultiVectorT<ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( aSpatialModel, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0 );
      }

      if( mBoundaryFluxes != nullptr )
      {
          mBoundaryFluxes->get( aSpatialModel, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0 );
      }
    }
};
// class ThermoelastostaticResidual

} // namespace Stabilized
} // namespace Plato
