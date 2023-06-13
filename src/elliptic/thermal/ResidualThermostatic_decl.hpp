#pragma once

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ApplyWeighting.hpp"
#include "ThermalConductivityMaterial.hpp"

#include "base/ResidualBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
class ResidualThermostatic : public Plato::ResidualBase
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

  using FunctionBaseType = Plato::ResidualBase;
  using FunctionBaseType::mSpatialDomain;
  using FunctionBaseType::mDataMap;
  using FunctionBaseType::mDofNames;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using GradScalarType    = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

  IndicatorFunctionType mIndicatorFunction;
  ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyWeighting;

  std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
  std::shared_ptr<Plato::NaturalBCs<ElementType, mNumDofsPerNode>> mBoundaryLoads;

  Teuchos::RCP<Plato::MaterialModel<EvaluationType>> mMaterialModel;

  std::vector<std::string> mPlottable;

public:
  ResidualThermostatic(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          Teuchos::ParameterList & penaltyParams
  );

  Plato::Solutions 
  getSolutionStateOutputData(
    const Plato::Solutions &aSolutions
  ) const;

  /// @fn evaluate
  /// @brief evaluate internal forces
  /// @param [in,out] aWorkSets domain and range workset database
  /// @param [in]     aCycle    scalar
  void
  evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) const;

  /// @fn evaluateBoundary
  /// @brief evaluate boundary forces
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        scalar
  void
  evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) const;

}; // class ResidualThermostatic

} // namespace Elliptic

} // namespace Plato
