#pragma once

#include "BodyLoads.hpp"
#include "ApplyWeighting.hpp"
#include "LinearElectroelasticMaterial.hpp"

#include "base/ResidualBase.hpp"
#include "bcs/neumann/NeumannBCs.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
class ElectroelastostaticResidual : public Plato::ResidualBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of stress-strain terms
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

  static constexpr Plato::OrdinalType NElecDims = 1;
  static constexpr Plato::OrdinalType NMechDims = mNumSpatialDims;  
  static constexpr Plato::OrdinalType EDofOffset = mNumSpatialDims;
  static constexpr Plato::OrdinalType MDofOffset = 0;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;

  IndicatorFunctionType mIndicatorFunction;
  ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyEDispWeighting;
  ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;

  std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;

  std::shared_ptr<Plato::NeumannBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
  std::shared_ptr<Plato::NeumannBCs<ElementType, NElecDims, mNumDofsPerNode, EDofOffset>> mBoundaryCharges;

  Teuchos::RCP<Plato::LinearElectroelasticMaterial<mNumSpatialDims>> mMaterialModel;

  std::vector<std::string> mPlottable;

public:
  ElectroelastostaticResidual(
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
  { return Plato::Elliptic::residual_t::LINEAR_ELECTRO_MECHANICAL; }

  /// @fn postProcess
  /// @brief post process solution database before output
  /// @param [in] aSolutions solution database
  void 
  postProcess(
    const Plato::Solutions &aSolutions
  );

  /// @brief evaluate electro-mechanical residual, internal forces only
  /// @param [in,out] aWorkSets domain and range workset database
  /// @param [in]     aCycle    scalar
  void 
  evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) const override;

  /// @brief evaluate boundary forces
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        scalar
  void 
  evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) const override;

};
// class ElectroelastostaticResidual

} // namespace Elliptic

} // namespace Plato
