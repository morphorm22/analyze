#ifndef THERMOELASTOSTATIC_RESIDUAL_HPP
#define THERMOELASTOSTATIC_RESIDUAL_HPP

#include <memory>

#include "PlatoTypes.hpp"
#include "FadTypes.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "GeneralStressDivergence.hpp"
#include "GeneralFluxDivergence.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "ApplyWeighting.hpp"
#include "InterpolateFromNodal.hpp"
#include "ThermoelasticMaterial.hpp"
#include "NaturalBCs.hpp"
#include "BodyLoads.hpp"
#include "BLAS2.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermoelastostaticResidual :
        public EvaluationType::ElementType,
        public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    static constexpr int NThrmDims = 1;
    static constexpr int NMechDims = mNumSpatialDims;

    static constexpr int TDofOffset = mNumSpatialDims;
    static constexpr int MDofOffset = 0;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyFluxWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;

    std::shared_ptr<Plato::NaturalBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>> mBoundaryFluxes;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

    std::vector<std::string> mPlottable;

public:
    /**************************************************************************/
    ThermoelastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyFluxWeighting   (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr),
        mBoundaryFluxes       (nullptr)
    /**************************************************************************/
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
        if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
        mDofNames.push_back("temperature");

        // create material model and get stiffness
        //
        Plato::ThermoelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
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
  
        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
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
    void evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::TMKinematics<ElementType>          kinematics;
      Plato::TMKinetics<ElementType>            kinetics(mMaterialModel);
      
      Plato::GeneralStressDivergence<ElementType, mNumDofsPerNode, MDofOffset> stressDivergence;
      Plato::GeneralFluxDivergence  <ElementType, mNumDofsPerNode, TDofOffset> fluxDivergence;

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> tCellTgrad("tgrad", tNumCells, mNumSpatialDims);
    
      Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tCellFlux("flux" , tNumCells, mNumSpatialDims);
    
      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyFluxWeighting  = mApplyFluxWeighting;
      Kokkos::parallel_for("compute element state", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumVoigtTerms,  GradScalarType>   tStrain(0.0);
          Plato::Array<ElementType::mNumSpatialDims, GradScalarType>   tTGrad (0.0);
          Plato::Array<ElementType::mNumVoigtTerms,  ResultScalarType> tStress(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tFlux  (0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          tVolume *= tCubWeights(iGpOrdinal);

          // compute strain and electric field
          //
          kinematics(iCellOrdinal, tStrain, tTGrad, aState, tGradient);
    
          // compute stress and electric displacement
          //
          StateScalarType tTemperature(0.0);
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          interpolateFromNodal(iCellOrdinal, tBasisValues, aState, tTemperature);
          kinetics(iCellOrdinal, tStress, tFlux, tStrain, tTGrad, tTemperature);

          // apply weighting
          //
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
          applyFluxWeighting  (iCellOrdinal, aControl, tBasisValues, tFlux);
    
          // compute divergence
          //
          stressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);
          fluxDivergence  (iCellOrdinal, aResult, tFlux,   tGradient, tVolume);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume*tStrain(i));
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
          }
          for(int i=0; i<ElementType::mNumSpatialDims; i++)
          {
              Kokkos::atomic_add(&tCellTgrad(iCellOrdinal,i), tVolume*tTGrad(i));
              Kokkos::atomic_add(&tCellFlux(iCellOrdinal,i), tVolume*tFlux(i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellStrain(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
          for(int i=0; i<ElementType::mNumSpatialDims; i++)
          {
              tCellTgrad(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellFlux(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
      }

      if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tCellStrain, "strain", mSpatialDomain);
      if( std::count(mPlottable.begin(),mPlottable.end(),"tgrad" ) ) toMap(mDataMap, tCellTgrad,  "tgrad",  mSpatialDomain);
      if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tCellStress, "stress", mSpatialDomain);
      if( std::count(mPlottable.begin(),mPlottable.end(),"flux"  ) ) toMap(mDataMap, tCellFlux,   "flux" ,  mSpatialDomain);

    }
    /**************************************************************************/
    void evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
      }

      if( mBoundaryFluxes != nullptr )
      {
          mBoundaryFluxes->get( aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
      }
    }
};
// class ThermoelastostaticResidual

} // namespace Elliptic

} // namespace Plato
#endif
