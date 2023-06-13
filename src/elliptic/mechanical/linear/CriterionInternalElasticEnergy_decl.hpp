#pragma once

#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class CriterionInternalElasticEnergy : public Plato::CriterionBase
{
private:
  /// @brief topologcial element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of voigt stress-strain terms 
  static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;

  using FunctionBaseType = typename Plato::CriterionBase;
  using FunctionBaseType::mSpatialDomain;
  using FunctionBaseType::mDataMap;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

  IndicatorFunctionType mIndicatorFunction;
  Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

  Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

public:
  /******************************************************************************//**
   * \brief Constructor
   * \param aSpatialDomain Plato Analyze spatial domain
   * \param aProblemParams input database for overall problem
   * \param aPenaltyParams input database for penalty function
  **********************************************************************************/
  CriterionInternalElasticEnergy(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aProblemParams,
            Teuchos::ParameterList & aPenaltyParams,
      const std::string            & aFunctionName
  );

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn evaluateConditional
  /// @brief evaluate criterion
  /// @param [in,out] aWorkSets function domain and range workset database
  /// @param [in]     aCycle    scalar 
  void
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;

}; // class CriterionInternalElasticEnergy

} // namespace Elliptic

} // namespace Plato
