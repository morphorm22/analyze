#pragma once

#include "elliptic/ThermostaticResidual_decl.hpp"

#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"

namespace Plato
{

namespace Elliptic
{

    template<typename EvaluationType, typename IndicatorFunctionType>
    ThermostaticResidual<EvaluationType, IndicatorFunctionType>::ThermostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & penaltyParams
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap),
        mIndicatorFunction (penaltyParams),
        mApplyWeighting    (mIndicatorFunction),
        mBodyLoads         (nullptr),
        mBoundaryLoads     (nullptr)
    /**************************************************************************/
    {
        // obligatory: define dof names in order
        mDofNames.push_back("temperature");

        Plato::ThermalConductionModelFactory<mNumSpatialDims> tMaterialFactory(aProblemParams);
        mMaterialModel = tMaterialFactory.create(aSpatialDomain.getMaterialName());

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
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType, mNumDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
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
    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Solutions ThermostaticResidual<EvaluationType, IndicatorFunctionType>::
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const 
    {
      return aSolutions;
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    ThermostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType>  computeGradient;
      Plato::ScalarGrad<ElementType>             scalarGrad;
      Plato::GeneralFluxDivergence<ElementType>  fluxDivergence;

      Plato::ThermalFlux<ElementType>            thermalFlux(mMaterialModel);

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType>   tCellGrad("temperature gradient", tNumCells, mNumSpatialDims);
      Plato::ScalarMultiVectorT<ResultScalarType> tCellFlux("thermal flux", tNumCells, mNumSpatialDims);

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> interpolateFromNodal;
    
      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyWeighting = mApplyWeighting;

      Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumSpatialDims, GradScalarType> tGrad(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tFlux(0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);
          auto tBasisValues = ElementType::basisValues(tCubPoint);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);
    
          scalarGrad(iCellOrdinal, tGrad, aState, tGradient);
    
          StateScalarType tTemperature(0.0);
          interpolateFromNodal(iCellOrdinal, tBasisValues, aState, tTemperature);
          thermalFlux(tFlux, tGrad, tTemperature);
    
          tVolume *= tCubWeights(iGpOrdinal);

          applyWeighting(iCellOrdinal, aControl, tBasisValues, tFlux);
    
          fluxDivergence(iCellOrdinal, aResult, tFlux, tGradient, tVolume, -1.0);
        
          for(int i=0; i<ElementType::mNumSpatialDims; i++)
          {
              Kokkos::atomic_add(&tCellGrad(iCellOrdinal,i), tVolume*tGrad(i));
              Kokkos::atomic_add(&tCellFlux(iCellOrdinal,i), tVolume*tFlux(i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumSpatialDims; i++)
          {
              tCellGrad(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellFlux(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
      }

      if( std::count(mPlottable.begin(),mPlottable.end(),"tgrad") ) toMap(mDataMap, tCellGrad, "tgrad", mSpatialDomain);
      if( std::count(mPlottable.begin(),mPlottable.end(),"flux" ) ) toMap(mDataMap, tCellFlux, "flux" , mSpatialDomain);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    ThermostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult,  1.0 );
        }
    }
} // namespace Elliptic

} // namespace Plato
