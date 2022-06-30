#pragma once

#include "elliptic/ElectroelastostaticResidual_decl.hpp"

#include "BLAS2.hpp"
#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "PlatoTypes.hpp"
#include "EMKinetics.hpp"
#include "EMKinematics.hpp"
#include "GradientMatrix.hpp"
#include "GeneralFluxDivergence.hpp"
#include "GeneralStressDivergence.hpp"

namespace Plato
{

namespace Elliptic
{

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::ElectroelastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyEDispWeighting  (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr),
        mBoundaryCharges      (nullptr)
    /**************************************************************************/
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
        if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
        mDofNames.push_back("electric potential");

        // create material model and get stiffness
        //
        Plato::ElectroelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
        mMaterialModel = mmfactory.create(mSpatialDomain.getMaterialName());

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
  
        // parse electrical boundary Conditions
        // 
        if(aProblemParams.isSublist("Electrical Natural Boundary Conditions"))
        {
            mBoundaryCharges = std::make_shared<Plato::NaturalBCs<ElementType, NElecDims, mNumDofsPerNode, EDofOffset>>
                (aProblemParams.sublist("Electrical Natural Boundary Conditions"));
        }
  
        auto tResidualParams = aProblemParams.sublist("Electroelastostatics");
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
    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Solutions ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
      return aSolutions;
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::EMKinematics<ElementType>          kinematics;
      Plato::EMKinetics<ElementType>            kinetics(mMaterialModel);
      
      Plato::GeneralStressDivergence<ElementType, mNumDofsPerNode, MDofOffset> stressDivergence;
      Plato::GeneralFluxDivergence  <ElementType, mNumDofsPerNode, EDofOffset> edispDivergence;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> tCellEField("efield", tNumCells, mNumSpatialDims);
    
      Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tCellEDisp ("edisp" , tNumCells, mNumSpatialDims);
    
      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyEDispWeighting  = mApplyEDispWeighting;
      Kokkos::parallel_for("compute element state", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumVoigtTerms,  GradScalarType>   tStrain(0.0);
          Plato::Array<ElementType::mNumSpatialDims, GradScalarType>   tEField(0.0);
          Plato::Array<ElementType::mNumVoigtTerms,  ResultScalarType> tStress(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tEDisp (0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          tVolume *= tCubWeights(iGpOrdinal);

          // compute strain and electric field
          //
          kinematics(iCellOrdinal, tStrain, tEField, aState, tGradient);
    
          // compute stress and electric displacement
          //
          kinetics(tStress, tEDisp, tStrain, tEField);

          // apply weighting
          //
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
          applyEDispWeighting (iCellOrdinal, aControl, tBasisValues, tEDisp);
    
          // compute divergence
          //
          stressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);
          edispDivergence (iCellOrdinal, aResult, tEDisp,  tGradient, tVolume);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume*tStrain(i));
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
          }
          for(int i=0; i<ElementType::mNumSpatialDims; i++)
          {
              Kokkos::atomic_add(&tCellEField(iCellOrdinal,i), tVolume*tEField(i));
              Kokkos::atomic_add(&tCellEDisp(iCellOrdinal,i), tVolume*tEDisp(i));
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
              tCellEField(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellEDisp(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
      }

     if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tCellStrain, "strain", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"efield") ) toMap(mDataMap, tCellEField, "efield", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tCellStress, "stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"edisp" ) ) toMap(mDataMap, tCellEDisp,  "edisp",  mSpatialDomain);

    }
    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {

        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
        }

        if( mBoundaryCharges != nullptr )
        {
            mBoundaryCharges->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
        }
    }
} // namespace Elliptic

} // namespace Plato
