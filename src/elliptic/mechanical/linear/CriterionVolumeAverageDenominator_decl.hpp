#pragma once

#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
class CriterionVolumeAverageDenominator : public Plato::CriterionBase
{
private:
  /// @brief local topological element typename
  using ElementType = typename EvaluationType::ElementType;

  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

  using CriterionBaseType = Plato::CriterionBase;
  using CriterionBaseType::mSpatialDomain;
  using CriterionBaseType::mDataMap;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  
  std::string mSpatialWeightFunction;

public:
  /// @brief class constructor
  /// @param aSpatialDomain contains mesh and model information
  /// @param aDataMap       output database
  /// @param aProblemParams input problem parameters
  /// @param aFunctionName  criterion function name
  CriterionVolumeAverageDenominator(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap, 
          Teuchos::ParameterList & aProblemParams, 
          std::string            & aFunctionName
  );

  /// @brief set spatial weight function
  /// @param aWeightFunctionString string
  void 
  setSpatialWeightFunction(
    std::string aWeightFunctionString
  ) override;

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  bool 
  isLinear() 
  const;
  
  /// @brief evaluate volume average denominator criterion
  /// @param [in] aWorkSets function domain and range workset database
  /// @param [in] aCycle    scalar 
  void
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;
};
// class CriterionVolumeAverageDenominator

} // namespace Elliptic

} // namespace Plato

