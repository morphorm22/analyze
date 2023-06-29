#pragma once

#include <memory>

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ApplyWeighting.hpp"
#include "ThermoelasticMaterial.hpp"

#include "base/ResidualBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
class ResidualThermoelastostatic :  public Plato::ResidualBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of stress-strain components
  static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;
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

  static constexpr int NThrmDims = 1;
  static constexpr int NMechDims = mNumSpatialDims;

  static constexpr int TDofOffset = mNumSpatialDims;
  static constexpr int MDofOffset = 0;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using GradScalarType    = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

  IndicatorFunctionType mIndicatorFunction;
  Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyFluxWeighting;
  Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;

  std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;

  std::shared_ptr<Plato::NaturalBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
  std::shared_ptr<Plato::NaturalBCs<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>> mBoundaryFluxes;

  Teuchos::RCP<Plato::MaterialModel<EvaluationType>> mMaterialModel;

  std::vector<std::string> mPlottable;

public:
  ResidualThermoelastostatic(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          Teuchos::ParameterList & aPenaltyParams
  );

  /// @fn type
  /// @brief get residual type
  /// @return residual_t enum
  Plato::Elliptic::residual_t
  type() 
  const
  { return Plato::Elliptic::residual_t::LINEAR_THERMO_MECHANICAL; }

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
  /// @brief evaluate boundary forces, pure virtual function
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        scalar
  void 
  evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) const;

}; // class ResidualThermoelastostatic

} // namespace Elliptic

} // namespace Plato
