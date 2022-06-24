#ifndef INTERNAL_ELECTROELASTIC_ENERGY_HPP
#define INTERNAL_ELECTROELASTIC_ENERGY_HPP

#include "elliptic/AbstractScalarFunction.hpp"

#include "LinearElectroelasticMaterial.hpp"
#include "FadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "ScalarProduct.hpp"
#include "EMKinematics.hpp"
#include "EMKinetics.hpp"
#include "ApplyWeighting.hpp"
#include "ToMap.hpp"
#include "GradientMatrix.hpp"
#include "elliptic/ExpInstMacros.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Compute internal electro-static energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalElectroelasticEnergy : 
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<mNumSpatialDims>> mMaterialModel;
    
    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;
    ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyEDispWeighting;

  public:
    /******************************************************************************//**
     * \brief Constructor
     * \param aSpatialDomain Plato Analyze spatial domain
     * \param aProblemParams input database for overall problem
     * \param aPenaltyParams input database for penalty function
    **********************************************************************************/
    InternalElectroelasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
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
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
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
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
};
// class InternalElectroelasticEnergy

} // namespace Elliptic

} // namespace Plato

#include "ElectromechanicsElement.hpp"

PLATO_ELLIPTIC_DEC(Plato::Elliptic::InternalElectroelasticEnergy, Plato::ElectromechanicsElement)

#endif
