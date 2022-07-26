#pragma once

#include "elliptic/hatching/MechanicsElement.hpp"
#include "elliptic/hatching/AbstractScalarFunction.hpp"

#include "FadTypes.hpp"
#include "SmallStrain.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "GradientMatrix.hpp"
#include "ElasticModelFactory.hpp"
#include "elliptic/hatching/LinearStress.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

/******************************************************************************//**
 * @brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalElasticEnergy : 
  public EvaluationType::ElementType,
  public Plato::Elliptic::Hatching::AbstractScalarFunction<EvaluationType>
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::Elliptic::Hatching::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using GlobalStateScalarType = typename EvaluationType::GlobalStateScalarType;
    using LocalStateScalarType  = typename EvaluationType::LocalStateScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

  public:
    /******************************************************************************//**
     * @brief Constructor
     * @param aSpatialDomain Plato Analyze spatial domain
     * @param aProblemParams input database for overall problem
     * @param aPenaltyParams input database for penalty function
    **********************************************************************************/
    InternalElasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::Hatching::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    {
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());
    }

    /******************************************************************************//**
     * @brief Evaluate internal elastic energy function
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarArray3DT     <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0) const override
    {
        using StrainScalarType = typename Plato::fad_type_t<ElementType, GlobalStateScalarType, ConfigScalarType>;
      
        auto tNumCells = mSpatialDomain.numCells();
      
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
        Plato::SmallStrain<ElementType>           tComputeVoigtStrainIncrement;
        Plato::ScalarProduct<mNumVoigtTerms>      tComputeScalarProduct;

        Plato::Elliptic::Hatching::LinearStress<ElementType> tComputeVoigtStress(mMaterialModel);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto tApplyWeighting  = mApplyWeighting;
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrainIncrement(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);
            tVolume *= tCubWeights(iGpOrdinal);

            // compute strain increment
            //
            tComputeVoigtStrainIncrement(iCellOrdinal, tStrainIncrement, aGlobalState, tGradient);

            // compute stress
            //
            tComputeVoigtStress(iCellOrdinal, iGpOrdinal, tStress, tStrainIncrement, aLocalState);

            // apply weighting
            //
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tApplyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
    
            // compute element internal energy (0.5 * inner product of total strain and weighted stress)
            //
            tComputeScalarProduct(iCellOrdinal, aResult, tStress, tStrainIncrement, tVolume, 0.5);

            Plato::Array<ElementType::mNumVoigtTerms, LocalStateScalarType> tLocalState;
            for(Plato::OrdinalType iVoigt=0; iVoigt<ElementType::mNumVoigtTerms; iVoigt++)
            {
                tLocalState(iVoigt) = aLocalState(iCellOrdinal, iGpOrdinal, iVoigt);
            }
            tComputeScalarProduct(iCellOrdinal, aResult, tStress, tLocalState, tVolume, 0.5);
        });
    }
};
// class InternalElasticEnergy

} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
