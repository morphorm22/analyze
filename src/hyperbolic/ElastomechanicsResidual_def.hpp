#pragma once

#include "hyperbolic/ElastomechanicsResidual_decl.hpp"

#include "ToMap.hpp"
#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"

#include "GradientMatrix.hpp"
#include "CellVolume.hpp"

#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "GeneralStressDivergence.hpp"

#include "ProjectToNode.hpp"
#include "InterpolateFromNodal.hpp"
#include "hyperbolic/RayleighStress.hpp"
#include "hyperbolic/InertialContent.hpp"

namespace Plato
{

namespace Hyperbolic
{
    template<typename EvaluationType, typename IndicatorFunctionType>
    TransientMechanicsResidual<EvaluationType, IndicatorFunctionType>::TransientMechanicsResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap,
                               {"displacement X", "displacement Y", "displacement Z"},
                               {"velocity X",     "velocity Y",     "velocity Z"    },
                               {"acceleration X", "acceleration Y", "acceleration Z"}),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyMassWeighting   (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr)
    {
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        mRayleighDamping = (mMaterialModel->getRayleighA() != 0.0)
                        || (mMaterialModel->getRayleighB() != 0.0);

        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>
                         (aProblemParams.sublist("Body Loads"));
        }

        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType>>
                             (aProblemParams.sublist("Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Hyperbolic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Scalar
    TransientMechanicsResidual<EvaluationType, IndicatorFunctionType>::
    getMaxEigenvalue(const Plato::ScalarArray3D & aConfig) const
    {
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVector tCellVolume("cell volume", tNumCells);

        Plato::ComputeCellVolume<ElementType> tComputeVolume;

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute cell volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            Plato::Scalar tVolume;
            auto tCubPoint = tCubPoints(iGpOrdinal);
            tComputeVolume(iCellOrdinal, tCubPoint, aConfig, tVolume);
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        Plato::Scalar tMinVolume;
        Plato::blas1::min(tCellVolume, tMinVolume);
        Plato::Scalar tLength = pow(tMinVolume, 1.0/mNumSpatialDims);

        auto tStiffnessMatrix = mMaterialModel->getStiffnessMatrix();
        auto tMassDensity     = mMaterialModel->getMassDensity();
        auto tSoundSpeed = sqrt(tStiffnessMatrix(0,0)/tMassDensity);

        return 2.0*tSoundSpeed/tLength;
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Solutions 
    TransientMechanicsResidual<EvaluationType, IndicatorFunctionType>::
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
      return aSolutions;
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    TransientMechanicsResidual<EvaluationType, IndicatorFunctionType>::
    evaluate(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep,
              Plato::Scalar aCurrentTime
    ) const
    {
        if ( mRayleighDamping )
        {
             evaluateWithDamping(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult, aTimeStep, aCurrentTime);
        }
        else
        {
             evaluateWithoutDamping(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult, aTimeStep, aCurrentTime);
        }
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    TransientMechanicsResidual<EvaluationType, IndicatorFunctionType>::
    evaluateWithoutDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep,
              Plato::Scalar aCurrentTime
    ) const
    {
      using StrainScalarType =
          typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::SmallStrain<ElementType>           computeVoigtStrain;
      Plato::LinearStress<EvaluationType, ElementType> computeVoigtStress(mMaterialModel);
      Plato::GeneralStressDivergence<ElementType> computeStressDivergence;

      Plato::InertialContent<ElementType>       computeInertialContent(mMaterialModel);
      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, /*offset=*/0, mNumSpatialDims> interpolateFromNodal;
      Plato::ProjectToNode<ElementType>         projectInertialContent;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("volume",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tCellStrain("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tCellStress("stress",tNumCells,mNumVoigtTerms);

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyMassWeighting = mApplyMassWeighting;

      Kokkos::parallel_for("compute residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);
          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrain(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

          Plato::Array<ElementType::mNumSpatialDims, StateDotDotScalarType> tAcceleration(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tInertialContent(0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          computeVoigtStrain(iCellOrdinal, tStrain, aState, tGradient);

          computeVoigtStress(tStress, tStrain);

          tVolume *= tCubWeights(iGpOrdinal);

          auto tBasisValues = ElementType::basisValues(tCubPoint);

          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

          computeStressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);
      
          interpolateFromNodal(iCellOrdinal, tBasisValues, aStateDotDot, tAcceleration);

          computeInertialContent(tInertialContent, tAcceleration);

          applyMassWeighting(iCellOrdinal, aControl, tBasisValues, tInertialContent);

          projectInertialContent(iCellOrdinal, tVolume, tBasisValues, tInertialContent, aResult);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume*tStrain(i));
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);

      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellStrain(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
      }

     if( std::count(mPlotTable.begin(),mPlotTable.end(),"stress") ) { Plato::toMap(mDataMap, tCellStress, "stress", mSpatialDomain); }
     if( std::count(mPlotTable.begin(),mPlotTable.end(),"strain") ) { Plato::toMap(mDataMap, tCellStrain, "strain", mSpatialDomain); }

    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    TransientMechanicsResidual<EvaluationType, IndicatorFunctionType>::
    evaluateWithDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep,
              Plato::Scalar aCurrentTime
    ) const
    {
      using StrainScalarType =
          typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
      using VelGradScalarType =
          typename Plato::fad_type_t<ElementType, StateDotScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::SmallStrain<ElementType>           computeVoigtStrain;
      Plato::RayleighStress<EvaluationType, ElementType> computeVoigtStress(mMaterialModel);
      Plato::GeneralStressDivergence<ElementType> computeStressDivergence;

      Plato::InertialContent<ElementType>        computeInertialContent(mMaterialModel);
      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, /*offset=*/0, mNumSpatialDims> interpolateFromNodal;
      Plato::ProjectToNode<ElementType>         projectInertialContent;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("volume",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tCellStrain("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<VelGradScalarType>
        tCellVelGrad("velgrad",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tCellStress("stress",tNumCells,mNumVoigtTerms);

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyMassWeighting = mApplyMassWeighting;

      Kokkos::parallel_for("compute residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);
          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrain(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, VelGradScalarType> tVelGrad(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

          Plato::Array<ElementType::mNumSpatialDims, StateDotDotScalarType> tAcceleration(0.0);
          Plato::Array<ElementType::mNumSpatialDims, StateDotScalarType> tVelocity(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tInertialContent(0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          computeVoigtStrain(iCellOrdinal, tStrain, aState, tGradient);

          computeVoigtStrain(iCellOrdinal, tVelGrad, aStateDot, tGradient);

          computeVoigtStress(tStress, tStrain, tVelGrad);

          tVolume *= tCubWeights(iGpOrdinal);

          auto tBasisValues = ElementType::basisValues(tCubPoint);

          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

          computeStressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);

          interpolateFromNodal(iCellOrdinal, tBasisValues, aStateDotDot, tAcceleration);

          interpolateFromNodal(iCellOrdinal, tBasisValues, aStateDot, tVelocity);

          computeInertialContent(tInertialContent, tVelocity, tAcceleration);
          
          applyMassWeighting(iCellOrdinal, aControl, tBasisValues, tInertialContent);

          projectInertialContent(iCellOrdinal, tVolume, tBasisValues, tInertialContent, aResult);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume*tStrain(i));
              Kokkos::atomic_add(&tCellVelGrad(iCellOrdinal,i), tVolume*tVelGrad(i));
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        
      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellStrain(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellVelGrad(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
      }

     if( std::count(mPlotTable.begin(),mPlotTable.end(),"stress") ) { Plato::toMap(mDataMap, tCellStress, "stress", mSpatialDomain); }
     if( std::count(mPlotTable.begin(),mPlotTable.end(),"velgrad") ) { Plato::toMap(mDataMap, tCellVelGrad, "velgrad", mSpatialDomain); }
     if( std::count(mPlotTable.begin(),mPlotTable.end(),"strain") ) { Plato::toMap(mDataMap, tCellStrain, "strain", mSpatialDomain); }

    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    TransientMechanicsResidual<EvaluationType, IndicatorFunctionType>::
    evaluate_boundary(
        const Plato::SpatialModel                                & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep,
              Plato::Scalar aCurrentTime
    ) const
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0, aCurrentTime );
        }
    }

}

}
