#pragma once

#include "elliptic/InternalElectroelasticEnergy_decl.hpp"

#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "EMKinetics.hpp"
#include "EMKinematics.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Constructor
     * \param aSpatialDomain Plato Analyze spatial domain
     * \param aProblemParams input database for overall problem
     * \param aPenaltyParams input database for penalty function
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    InternalElectroelasticEnergy<EvaluationType, IndicatorFunctionType>::InternalElectroelasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyEDispWeighting  (mIndicatorFunction)
    {
      Plato::ElectroelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
    }

    /******************************************************************************//**
     * \brief Evaluate internal elastic energy function
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    InternalElectroelasticEnergy<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
      Plato::EMKinematics<ElementType>          tKinematics;
      Plato::EMKinetics<ElementType>            tKinetics(mMaterialModel);

      Plato::ScalarProduct<mNumVoigtTerms>     tMechanicalScalarProduct;
      Plato::ScalarProduct<mNumSpatialDims>    tElectricalScalarProduct;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyEDispWeighting  = mApplyEDispWeighting;
      Kokkos::parallel_for("compute internal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<mNumVoigtTerms,  GradScalarType>   tStrain(0.0);
          Plato::Array<mNumSpatialDims, GradScalarType>   tEField(0.0);
          Plato::Array<mNumVoigtTerms,  ResultScalarType> tStress(0.0);
          Plato::Array<mNumSpatialDims, ResultScalarType> tEDisp (0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          tVolume *= tCubWeights(iGpOrdinal);

          // compute strain and electric field
          //
          tKinematics(iCellOrdinal, tStrain, tEField, aState, tGradient);

          // compute stress and electric displacement
          //
          tKinetics(tStress, tEDisp, tStrain, tEField);

          // apply weighting
          //
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          tApplyStressWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
          tApplyEDispWeighting (iCellOrdinal, aControl, tBasisValues, tEDisp);

          // compute element internal energy
          //
          tMechanicalScalarProduct(iCellOrdinal, aResult, tStress, tStrain, tVolume);
          tElectricalScalarProduct(iCellOrdinal, aResult, tEDisp,  tEField, tVolume, -1.0);
      });
    }
} // namespace Elliptic

} // namespace Plato
