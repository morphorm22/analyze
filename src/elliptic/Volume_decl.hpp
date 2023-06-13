#pragma once

#include "SpatialModel.hpp"
#include "ApplyWeighting.hpp"
#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename PenaltyFunctionType>
class Volume : public Plato::CriterionBase
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
  /// @brief contains mesh and model information
  using Plato::CriterionBase::mSpatialDomain;
  /// @brief output database
  using Plato::CriterionBase::mDataMap;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  /// @brief interface to density-based penalty function
  PenaltyFunctionType mPenaltyFunction;
  /// @brief interface to apply density-based penalization
  Plato::ApplyWeighting<mNumNodesPerCell, /*num_weighted_terms=*/1, PenaltyFunctionType> mApplyWeighting;

public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  /// @param [in] aProblemParams input problem parameters
  /// @param [in] aPenaltyParams input density-based penalty function parameters
  /// @param [in] aFunctionName  criterion parameter list name
  Volume(
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
};
// class Volume

} // namespace Elliptic

} // namespace Plato
