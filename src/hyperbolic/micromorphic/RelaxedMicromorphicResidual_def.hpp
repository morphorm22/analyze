#pragma once

#include "hyperbolic/micromorphic/RelaxedMicromorphicResidual_decl.hpp"

#include "ToMap.hpp"
#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"

#include "GradientMatrix.hpp"
#include "CellVolume.hpp"

#include "ProjectToNode.hpp"
#include "InterpolateFromNodal.hpp"

#include "hyperbolic/InertialContent.hpp"

#include "hyperbolic/micromorphic/MicromorphicKinematics.hpp"
#include "hyperbolic/micromorphic/MicromorphicKinetics.hpp"
#include "hyperbolic/micromorphic/FullStressDivergence.hpp"
#include "hyperbolic/micromorphic/ProjectStressToNode.hpp"

namespace Plato
{

namespace Hyperbolic
{

    template<typename EvaluationType, typename IndicatorFunctionType>
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::RelaxedMicromorphicResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap,
                               {"Displacement X", "Displacement Y", "Displacement Z",
                               "Micro Distortion XX", "Micro Distortion YX", "Micro Distortion ZX",
                               "Micro Distortion XY", "Micro Distortion YY", "Micro Distortion ZY",
                               "Micro Distortion XZ", "Micro Distortion YZ", "Micro Distortion ZZ"},
                               {"Velocity X", "Velocity Y", "Velocity Z",
                               "Micro Velocity XX", "Micro Velocity YX", "Micro Velocity ZX",
                               "Micro Velocity XY", "Micro Velocity YY", "Micro Velocity ZY",
                               "Micro Velocity XZ", "Micro Velocity YZ", "Micro Velocity ZZ"},
                               {"Acceleration X", "Acceleration Y", "Acceleration Z",
                               "Micro Acceleration XX", "Micro Acceleration YX", "Micro Acceleration ZX",
                               "Micro Acceleration XY", "Micro Acceleration YY", "Micro Acceleration ZY",
                               "Micro Acceleration XZ", "Micro Acceleration YZ", "Micro Acceleration ZZ"}),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyMassWeighting   (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr)
    {
        this->checkTimeIntegrator(aProblemParams.sublist("Time Integration"));

        Plato::MicromorphicElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        Plato::MicromorphicInertiaModelFactory<mNumSpatialDims> tInertiaModelFactory(aProblemParams);
        mInertiaModel = tInertiaModelFactory.create(aSpatialDomain.getMaterialName());

        mRayleighDamping = (mInertiaModel->getRayleighA() != 0.0)
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
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    checkTimeIntegrator(Teuchos::ParameterList & aIntegratorParams)
    {
        if (aIntegratorParams.isType<bool>("A-Form"))
        {
            auto tAForm = aIntegratorParams.get<bool>("A-Form");
            auto tBeta = aIntegratorParams.get<double>("Newmark Beta");
            if (tAForm == false)
            {
                ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark A-Form must be specified for micromorphic mechanics")
            }
            else if (tBeta != 0.0)
            {
                ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark explicit (beta=0, gamma=0.5) must be specified for micromorphic mechanics")
            }
        }
        else
        {
            ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark A-Form must be specified for micromorphic mechanics")
        }
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Scalar
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    getMaxEigenvalue(const Plato::ScalarArray3D & aConfig) const
    {
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVector tCellVolume("cell weight", tNumCells);

        Plato::ComputeCellVolume<ElementType> tComputeVolume;

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute cell volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            Plato::Scalar tVolume;
            auto tCubPoint = tCubPoints(iGpOrdinal);
            tComputeVolume(iCellOrdinal, tCubPoint, aConfig, tVolume);
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        Plato::Scalar tMinVolume;
        Plato::blas1::min(tCellVolume, tMinVolume);
        Plato::Scalar tLength = pow(tMinVolume, 1.0/mNumSpatialDims);

        auto tStiffnessMatrixCe = mMaterialModel->getStiffnessMatrixCe();
        auto tMassDensity     = mInertiaModel->getMacroscopicMassDensity();
        auto tSoundSpeed = sqrt(tStiffnessMatrixCe(0,0)/tMassDensity);

        return 2.0*tSoundSpeed/tLength;
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Solutions 
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
      return aSolutions;
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
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
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
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

      using InertiaStrainScalarType =
          typename Plato::fad_type_t<ElementType, StateDotDotScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::Hyperbolic::MicromorphicKinematics<ElementType> computeKinematics;
      Plato::Hyperbolic::MicromorphicKinetics<ElementType>   computeKinetics(mMaterialModel);
      Plato::Hyperbolic::MicromorphicKinetics<ElementType>   computeInertiaKinetics(mInertiaModel);
      Plato::Hyperbolic::FullStressDivergence<ElementType>   computeFullStressDivergence;
      Plato::InertialContent<ElementType>                    computeInertialContent(mInertiaModel);
      Plato::ProjectToNode<ElementType, mNumSpatialDims>                      projectInertialContent;
      Plato::Hyperbolic::ProjectStressToNode<ElementType>    computeStressForMicromorphicResidual;
      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, /*offset=*/0, mNumSpatialDims> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType> 
        tCellSymDisplacementGradient("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tCellSymCauchyStress("stress",tNumCells,mNumVoigtTerms);

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyMassWeighting = mApplyMassWeighting;

      Kokkos::parallel_for("compute residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);
          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tSymDisplacementGradient(0.0);
          Plato::Array<ElementType::mNumSkwTerms, StrainScalarType>   tSkwDisplacementGradient(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, StateScalarType>  tSymMicroDistortionTensor(0.0);
          Plato::Array<ElementType::mNumSkwTerms, StateScalarType>    tSkwMicroDistortionTensor(0.0);

          Plato::Array<ElementType::mNumVoigtTerms, InertiaStrainScalarType> tSymGradientMicroInertia(0.0);
          Plato::Array<ElementType::mNumSkwTerms, InertiaStrainScalarType>   tSkwGradientMicroInertia(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, StateDotDotScalarType>   tSymFreeMicroInertia(0.0);
          Plato::Array<ElementType::mNumSkwTerms, StateDotDotScalarType>     tSkwFreeMicroInertia(0.0);

          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tSymCauchyStress(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tSkwCauchyStress(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tSymMicroStress(0.0);
        
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tSymGradientInertiaStress(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tSkwGradientInertiaStress(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tSymFreeInertiaStress(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tSkwFreeInertiaStress(0.0);

          Plato::Array<ElementType::mNumSpatialDims, StateDotDotScalarType> tAcceleration(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType>      tInertialContent(0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);
          auto tBasisValues = ElementType::basisValues(tCubPoint);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);
         
          tVolume *= tCubWeights(iGpOrdinal);

          computeKinematics(iCellOrdinal, tSymDisplacementGradient, tSkwDisplacementGradient, tSymMicroDistortionTensor, tSkwMicroDistortionTensor, aState, tBasisValues, tGradient);
          computeKinematics(iCellOrdinal, tSymGradientMicroInertia, tSkwGradientMicroInertia, tSymFreeMicroInertia, tSkwFreeMicroInertia, aStateDotDot, tBasisValues, tGradient);

          computeKinetics(tSymCauchyStress, tSkwCauchyStress, tSymMicroStress, tSymDisplacementGradient, tSkwDisplacementGradient, tSymMicroDistortionTensor, tSkwMicroDistortionTensor);
          computeInertiaKinetics(tSymGradientInertiaStress, tSkwGradientInertiaStress, tSymFreeInertiaStress, tSkwFreeInertiaStress, tSymGradientMicroInertia, tSkwGradientMicroInertia, tSymFreeMicroInertia, tSkwFreeMicroInertia);

          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tSymCauchyStress);
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tSkwCauchyStress);
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tSymMicroStress);
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tSymGradientInertiaStress);
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tSkwGradientInertiaStress);
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tSymFreeInertiaStress);
          applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tSkwFreeInertiaStress);

        //   computeFullStressDivergence(iCellOrdinal, aResult, tSymCauchyStress, tSkwCauchyStress, tGradient, tVolume);
        //   computeStressForMicromorphicResidual(iCellOrdinal, aResult, tSymCauchyStress, tSkwCauchyStress, tSymMicroStress, tBasisValues, tVolume);

        //   computeFullStressDivergence(iCellOrdinal, aResult, tSymGradientInertiaStress, tSkwGradientInertiaStress, tGradient, tVolume);
        //   computeStressForMicromorphicResidual(iCellOrdinal, aResult, tSymFreeInertiaStress, tSkwFreeInertiaStress, tBasisValues, tVolume);

          interpolateFromNodal(iCellOrdinal, tBasisValues, aStateDotDot, tAcceleration);
          computeInertialContent(tInertialContent, tAcceleration);
          applyMassWeighting(iCellOrdinal, aControl, tBasisValues, tInertialContent);
          projectInertialContent(iCellOrdinal, tVolume, tBasisValues, tInertialContent, aResult);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellSymDisplacementGradient(iCellOrdinal,i), tVolume*tSymDisplacementGradient(i));
              Kokkos::atomic_add(&tCellSymCauchyStress(iCellOrdinal,i), tVolume*tSymCauchyStress(i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);

      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellSymDisplacementGradient(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellSymCauchyStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
      }

     if( std::count(mPlotTable.begin(),mPlotTable.end(),"stress") ) { Plato::toMap(mDataMap, tCellSymCauchyStress, "stress", mSpatialDomain); }
     if( std::count(mPlotTable.begin(),mPlotTable.end(),"strain") ) { Plato::toMap(mDataMap, tCellSymDisplacementGradient, "strain", mSpatialDomain); }

    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
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
        ANALYZE_THROWERR("Relaxed Micromorphic residual does not support damping currently.")
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
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

