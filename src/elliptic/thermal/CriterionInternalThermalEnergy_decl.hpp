#pragma once

#include "ApplyWeighting.hpp"

#include "base/CriterionBase.hpp"
#include "materials/MaterialModel.hpp"
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
class CriterionInternalThermalEnergy : public Plato::CriterionBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of integration points
  static constexpr auto mNumGaussPoints  = ElementType::mNumGaussPoints;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;

  using Plato::CriterionBase::mSpatialDomain; 
  using Plato::CriterionBase::mDataMap; 

  using StateScalarType   = typename EvaluationType::StateScalarType; 
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType; 
  using GradScalarType    = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

  IndicatorFunctionType mIndicatorFunction; /*!< penalty function */
  Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyWeighting; /*!< applies penalty function */
  
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:
  /// @brief class constructor
  /// @param aSpatialDomain contains mesh and model information
  /// @param aDataMap       output database
  /// @param aProblemParams input problem parameters
  /// @param aPenaltyParams input penalty model parameters
  /// @param aFunctionName  criterion parameter list name
  CriterionInternalThermalEnergy(
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
  void evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;

};
// class CriterionInternalThermalEnergy

} // namespace Elliptic

} // namespace Plato
