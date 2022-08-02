#pragma once

#include "PlatoUtilities.hpp"

#include <memory>

#include "stabilized/Kinetics.hpp"
#include "stabilized/Kinematics.hpp"
#include "stabilized/Projection.hpp"
#include "stabilized/MechanicsElement.hpp"
#include "stabilized/AbstractVectorFunction.hpp"
#include "PlatoTypes.hpp"
#include "GeneralFluxDivergence.hpp"
#include "GeneralStressDivergence.hpp"
#include "PressureDivergence.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "CellForcing.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ElasticModelFactory.hpp"
#include "NaturalBCs.hpp"
#include "BodyLoads.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************//**
 * \brief Stabilized elastostatic residual (reference: M. Chiumenti et al. (2004))
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticResidual :
        public EvaluationType::ElementType,
        public Plato::Stabilized::AbstractVectorFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumSpatialDims;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumMechDims  = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumPressDims = 1;
    static constexpr Plato::OrdinalType mMechDofOffset = 0;
    static constexpr Plato::OrdinalType mPressDofOffset = mNumSpatialDims;

    using FunctionBaseType = Plato::Stabilized::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction; /*!< material penalty function */
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyTensorWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyVectorWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, 1 /* number of pressure dofs per node */, IndicatorFunctionType> mApplyScalarWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads; /*!< body loads interface */
    std::shared_ptr<Plato::NaturalBCs<ElementType, mNumMechDims, mNumDofsPerNode, mMechDofOffset>> mBoundaryLoads; /*!< boundary loads interface */

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel; /*!< material model interface */

    std::vector<std::string> mPlottable; /*!< array with output data identifiers */

private:
    /******************************************************************************//**
     * \brief initialize material, loads and output data
     * \param [in] aProblemParams input XML data
    **********************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams)
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
        if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
        mDofNames.push_back("pressure");

        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialFactory(aProblemParams);
        mMaterialModel = tMaterialFactory.create(mSpatialDomain.getMaterialName());
  

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
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType, mNumMechDims, mNumDofsPerNode, mMechDofOffset>>
                                (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
  
        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
          mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
     * \param [in] aDataMap output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
    **********************************************************************************/
    ElastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyTensorWeighting (mIndicatorFunction),
        mApplyVectorWeighting (mIndicatorFunction),
        mApplyScalarWeighting (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr)
    {
        this->initialize(aProblemParams);
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

    /******************************************************************************//**
     * \brief Evaluate stabilized elastostatics residual
     * \param [in] aStateWS state, {disp_x, disp_y, disp_z, pressure}, workset
     * \param [in] aPressGradWS pressure gradient workset
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual, workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <NodeStateScalarType> & aPressGradWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarMultiVectorT <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight", tNumCells);
      Plato::ScalarVectorT<ResultScalarType> tCellPressure("cell pressure", tNumCells);
      Plato::ScalarMultiVectorT<ResultScalarType> tCellDevStress("stress", tNumCells, mNumVoigtTerms);

      Plato::ComputeGradientMatrix  <ElementType> tComputeGradient;
      Plato::Stabilized::Kinematics <ElementType> tKinematics;
      Plato::Stabilized::Kinetics   <ElementType> tKinetics(mMaterialModel);

      Plato::InterpolateFromNodal   <ElementType, mNumDofsPerNode, mPressDofOffset>  tInterpolatePressureFromNodal;
      Plato::InterpolateFromNodal   <ElementType, mNumSpatialDims, /*dof_offset=*/0, mNumSpatialDims> tInterpolatePGradFromNodal;
      
      Plato::PressureDivergence      <ElementType, mNumDofsPerNode>                  tPressureDivergence;
      Plato::GeneralStressDivergence <ElementType, mNumDofsPerNode, mMechDofOffset>  tStressDivergence;
      Plato::GeneralFluxDivergence   <ElementType, mNumDofsPerNode, mPressDofOffset> tStabilizedDivergence;
      Plato::ProjectToNode           <ElementType, mNumDofsPerNode, mPressDofOffset> tProjectVolumeStrain;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& tApplyTensorWeighting = mApplyTensorWeighting;
      auto& tApplyVectorWeighting = mApplyVectorWeighting;
      auto& tApplyScalarWeighting = mApplyScalarWeighting;

      Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        ConfigScalarType tVolume(0.0);

        Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;


        // compute gradient operator and cell volume
        //
        auto tCubPoint = tCubPoints(iGpOrdinal);
        tComputeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
        tVolume *= tCubWeights(iGpOrdinal);

        // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
        //
        Plato::Array<mNumVoigtTerms, GradScalarType> tDGrad(0.0);
        Plato::Array<mNumSpatialDims, GradScalarType> tPGrad(0.0);
        tKinematics(iCellOrdinal, tDGrad, tPGrad, aStateWS, tGradient);

        // interpolate projected PGrad, pressure, and temperature to gauss point
        //
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        Plato::Array<mNumSpatialDims, NodeStateScalarType> tProjectedPGrad(0.0);
        tInterpolatePGradFromNodal(iCellOrdinal, tBasisValues, aPressGradWS, tProjectedPGrad);

        ResultScalarType tPressure;
        tInterpolatePressureFromNodal(iCellOrdinal, tBasisValues, aStateWS, tPressure);

        // compute the constitutive response
        //
        ResultScalarType tVolStrain(0.0);
        Plato::Array<mNumSpatialDims, ResultScalarType> tCellStab(0.0);
        Plato::Array<mNumVoigtTerms, ResultScalarType> tDevStress(0.0);
        tKinetics(tVolume, tProjectedPGrad, tDGrad, tPGrad,
                  tPressure, tDevStress, tVolStrain, tCellStab);

        // apply weighting
        //
        tApplyTensorWeighting (iCellOrdinal, aControlWS, tBasisValues, tDevStress);
        tApplyVectorWeighting (iCellOrdinal, aControlWS, tBasisValues, tCellStab);
        tApplyScalarWeighting (iCellOrdinal, aControlWS, tBasisValues, tPressure);
        tApplyScalarWeighting (iCellOrdinal, aControlWS, tBasisValues, tVolStrain);
    
        // compute divergence
        //
        tStressDivergence    (iCellOrdinal, aResultWS,  tDevStress, tGradient, tVolume);
        tPressureDivergence  (iCellOrdinal, aResultWS,  tPressure,  tGradient, tVolume);
        tStabilizedDivergence(iCellOrdinal, aResultWS,  tCellStab,  tGradient, tVolume, -1.0);

        tProjectVolumeStrain (iCellOrdinal, tVolume, tBasisValues, tVolStrain, aResultWS);

        for(int i=0; i<mNumVoigtTerms; i++)
        {
            Kokkos::atomic_add(&tCellDevStress(iCellOrdinal,i), tVolume*tDevStress(i));
        }
        Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        Kokkos::atomic_add(&tCellPressure(iCellOrdinal), tPressure);
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0 );
      }

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellDevStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
          tCellPressure(iCellOrdinal) /= tCellVolume(iCellOrdinal);
      });

      if(std::count(mPlottable.begin(),mPlottable.end(), "pressure"         )) {toMap(mDataMap, tCellPressure,  "pressure", mSpatialDomain); }
      if(std::count(mPlottable.begin(),mPlottable.end(), "deviatoric stress")) {toMap(mDataMap, tCellDevStress, "deviatoric stress", mSpatialDomain); }
    }
    /******************************************************************************//**
     * \brief Evaluate stabilized elastostatics boundary terms residual
     * \param [in] aStateWS state, {disp_x, disp_y, disp_z, pressure}, workset
     * \param [in] aPressGradWS pressure gradient workset
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual, workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                             & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <NodeStateScalarType> & aPressGradWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarMultiVectorT <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( aSpatialModel, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0 );
      }
    }
};
// class ElastostaticResidual

} // namespace Stabilized
} // namespace Plato
