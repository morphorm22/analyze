/*
 * StressEvaluator.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/// @class StressEvaluator
/// @brief base class for stress evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluator
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;

protected:
  /// @brief contains mesh and model information
  const Plato::SpatialDomain & mSpatialDomain;
  /// @brief output database 
  Plato::DataMap & mDataMap;

public:
  /// @brief base class constructor
  StressEvaluator(){}

  /// @brief base class construtor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  explicit
  StressEvaluator(
      const Plato::SpatialDomain & aSpatialDomain,
            Plato::DataMap       & aDataMap
  ) :
    mSpatialDomain(aSpatialDomain),
    mDataMap(aDataMap)
  {}

  /// @brief base class destructor
  virtual ~StressEvaluator(){}

  /// @fn evaluate
  /// @brief evaluate stress tensor - volume integral use case
  /// @param [in]     aWorkSets domain and range workset database
  /// @param [in,out] aResult   4D scalar container
  /// @param [in]     aCycle    scalar
  virtual 
  void 
  evaluate(
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle = 0.0
  ) const = 0;

};

}