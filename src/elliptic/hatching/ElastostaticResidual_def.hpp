#pragma once

#include "ToMap.hpp"
#include "PlatoTypes.hpp"
#include "SmallStrain.hpp"
#include "CellForcing.hpp"
#include "GradientMatrix.hpp"
#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "GeneralStressDivergence.hpp"
#include "elliptic/hatching/LinearStress.hpp"
#include "elliptic/mechanical/linear/VonMisesYieldFunction.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze database
     * \param [in] aProblemParams input parameters for overall problem
     * \param [in] aPenaltyParams input parameters for penalty function
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    ElastostaticResidual<EvaluationType, IndicatorFunctionType>::
    ElastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction),
        mBodyLoads         (nullptr),
        mBoundaryLoads     (nullptr)
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
        if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");

        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType>>(aProblemParams.sublist("Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Updated Lagrangian Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
          mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }

    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Solutions
    ElastostaticResidual<EvaluationType, IndicatorFunctionType>::
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
      Plato::ScalarMultiVector tDisplacements = aSolutions.get("State");
      Plato::Solutions tSolutionsOutput(aSolutions.physics(), aSolutions.pde());
      tSolutionsOutput.set("Displacement", tDisplacements, mDofNames);
      return tSolutionsOutput;
    }
    
    /******************************************************************************//**
     * \brief Evaluate vector function
     *
     * \param [in] aGlobalState 2D array with global state variables (C,DOF)
     * \param [in] aLocalState 2D array with local state variables (C, NS)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
     * NS = number of local states per cell
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    ElastostaticResidual<EvaluationType, IndicatorFunctionType>::
    evaluate(
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarArray3DT     <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep
    ) const
    {
      using StrainScalarType = typename Plato::fad_type_t<ElementType, GlobalStateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType>   tComputeGradient;
      Plato::SmallStrain<ElementType>             tComputeVoigtStrainIncrement;
      Plato::GeneralStressDivergence<ElementType> tComputeStressDivergence;

      Plato::Elliptic::Hatching::LinearStress<ElementType> tComputeVoigtStress(mMaterialModel);

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType> tCellStrainIncrement("strain increment", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      Plato::ScalarArray3DT<StrainScalarType> tGPStrainIncrement("strain increment", tNumCells, tNumPoints, mNumVoigtTerms);

      auto& applyWeighting = mApplyWeighting;

      Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrainIncrement(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);
          tVolume *= tCubWeights(iGpOrdinal);

          // compute strain increment
          tComputeVoigtStrainIncrement(iCellOrdinal, tStrainIncrement, aGlobalState, tGradient);
    
          // compute stress
          tComputeVoigtStress(iCellOrdinal, iGpOrdinal, tStress, tStrainIncrement, aLocalState);

          // apply weighting
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          applyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
    
          // compute stress divergence
          tComputeStressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tGPStrainIncrement(iCellOrdinal, iGpOrdinal, i) = tStrainIncrement(i);

              Kokkos::atomic_add(&tCellStrainIncrement(iCellOrdinal,i), tVolume*tStrainIncrement(i));
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellStrainIncrement(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aGlobalState, aControl, aConfig, aResult, -1.0 );
      }

      // Required in DataMap for state update
      Plato::toMap(mDataMap, tGPStrainIncrement, "strain increment", mSpatialDomain);

      if(std::count(mPlotTable.begin(), mPlotTable.end(), "stress")) { Plato::toMap(mDataMap, tCellStress, "stress", mSpatialDomain); }
      if(std::count(mPlotTable.begin(), mPlotTable.end(), "vonmises")) { this->outputVonMises(tCellStress, mSpatialDomain); }
    }
    /******************************************************************************//**
     * \brief Evaluate vector function
     *
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aGlobalState 2D array with state variables (C,DOF)
     * \param [in] aLocalState 2D array with local state variables (C, NS)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
     * NS = number of local states
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    ElastostaticResidual<EvaluationType, IndicatorFunctionType>::
    evaluate_boundary(
        const Plato::SpatialModel                               & aSpatialModel,
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarArray3DT     <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep
    ) const
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aGlobalState, aControl, aConfig, aResult, -1.0 );
        }
    }


    /**********************************************************************//**
     * \brief Compute Von Mises stress field and copy data into output data map
     * \param [in] aCauchyStress Cauchy stress tensor
    **************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    ElastostaticResidual<EvaluationType, IndicatorFunctionType>::
    outputVonMises(
        const Plato::ScalarMultiVectorT<ResultScalarType> & aCauchyStress,
        const Plato::SpatialDomain                        & aSpatialDomain
    ) const
    {
            auto tNumCells = aSpatialDomain.numCells();
            Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;
            Plato::ScalarVectorT<ResultScalarType> tVonMises("Von Mises", tNumCells);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
            {
                ResultScalarType tCellVonMises(0);
                tComputeVonMises(aCellOrdinal, aCauchyStress, tCellVonMises);
                tVonMises(aCellOrdinal) = tCellVonMises;
            }, "Compute VonMises Stress");

            Plato::toMap(mDataMap, tVonMises, "vonmises", aSpatialDomain);
    }
} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
