#pragma once

#include "ApplyWeighting.hpp"
#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Compute internal effective energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class CriterionEffectiveEnergy : public Plato::CriterionBase
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

  using Plato::CriterionBase::mSpatialDomain;
  using Plato::CriterionBase::mDataMap;
  using Plato::CriterionBase::mFunctionName;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

  IndicatorFunctionType mIndicatorFunction;
  Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

  Plato::Matrix< mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;
  Plato::Array<mNumVoigtTerms> mAssumedStrain;
  Plato::OrdinalType mColumnIndex;

  std::vector<std::string> mPlottable;

public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  /// @param [in] aProblemParams input problem parameters
  /// @param [in] aPenaltyParams input density-based penalty function parameters
  /// @param [in] aFunctionName  criterion parameter list name
  CriterionEffectiveEnergy(
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
  /// @brief evaluate effective internal energy criterion
  /// @param [in,out] aWorkSets function domain and range workset database
  /// @param [in]     aCycle    scalar 
  void 
  evaluateConditional(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
  ) const;

}; // class CriterionEffectiveEnergy

} // namespace Elliptic

} // namespace Plato
