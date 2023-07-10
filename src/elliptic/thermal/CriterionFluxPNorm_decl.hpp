#pragma once

#include "MaterialModel.hpp"
#include "ApplyWeighting.hpp"

#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
class CriterionFluxPNorm : public Plato::CriterionBase
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
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using GradScalarType    = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
  
  IndicatorFunctionType mIndicatorFunction;
  
  Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyWeighting;
  
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;
  
  Plato::OrdinalType mExponent;

public:
  CriterionFluxPNorm(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap, 
          Teuchos::ParameterList & aProblemParams, 
          Teuchos::ParameterList & aPenaltyParams,
          std::string              aFunctionName
  );

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn evaluateConditional
  /// @brief evaluate flux p-norm criterion
  /// @param [in,out] aWorkSets function domain and range workset database
  /// @param [in]     aCycle    scalar 
  void evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;

  void
  postEvaluate( 
    Plato::ScalarVector resultVector,
    Plato::Scalar       resultScalar);

  void
  postEvaluate( 
    Plato::Scalar & resultValue 
  );

};
// class CriterionFluxPNorm

} // namespace Elliptic

} // namespace Plato
