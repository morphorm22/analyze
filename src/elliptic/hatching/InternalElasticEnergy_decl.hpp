#pragma once

#include "FadTypes.hpp"
#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "elliptic/hatching/AbstractScalarFunction.hpp"

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
        const std::string            & aFunctionName
    );

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
              Plato::Scalar aTimeStep = 0.0) const override;
};
// class InternalElasticEnergy

} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
