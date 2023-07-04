#pragma once

#include "ApplyWeighting.hpp"
#include "ThermalConductivityMaterial.hpp"

#include "base/ResidualBase.hpp"
#include "bcs/body/BodyLoads.hpp"
#include "bcs/neumann/NeumannBCs.hpp"
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

  /// @brief volumetric heat source
  std::shared_ptr<Plato::NeumannBCs<EvaluationType>> mBoundaryForces;
  std::shared_ptr<Plato::BodyLoads<EvaluationType>> mHeatSource;

  Teuchos::RCP<Plato::MaterialModel<EvaluationType>> mMaterialModel;

  std::vector<std::string> mPlottable;

public:
  ResidualThermostatic(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          Teuchos::ParameterList & penaltyParams
  );

  /// @fn type
  /// @brief get residual type
  /// @return residual_t enum
  Plato::Elliptic::residual_t
  type() 
  const
  { return Plato::Elliptic::residual_t::LINEAR_THERMAL; }

  /// @fn postProcess
  /// @brief post process solution database before output
  /// @param [in] aSolutions solution database
  void
  postProcess(
    const Plato::Solutions &aSolutions
  );

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

private:
  /// @fn parseHeatSource
  /// @brief parse volumetric heat source
  /// @param [in] aProblemParams input problem parameters
  void
  parseHeatSource(
    Teuchos::ParameterList & aProblemParams
  );

  /// @fn parseNeumannBCs
  /// @brief parse thermal natural boundary conditions
  /// @param [in] aProblemParams input problem parameters
  void
  parseNeumannBCs(
    Teuchos::ParameterList & aProblemParams
  );

  /// @fn parseOutputs
  /// @brief parse thermal natural boundary conditions
  /// @param [in] aProblemParams input problem parameters
  void
  parseOutputs(
    Teuchos::ParameterList & aProblemParams
  );

}; // class ResidualThermostatic

} // namespace Elliptic

} // namespace Plato
