/*
 * CurrentDensityEvaluator.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "PlatoTypes.hpp"

namespace Plato
{

/// @brief base class for current density evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class CurrentDensityEvaluator
{
private:
    /// @brief scalar types for an evaluation type
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

public:
  /// @fn evaluate
  /// @brief pure virtual method: evaluates current density 
  /// @param aSpatialDomain contains meshed model information
  /// @param aState         state workset
  /// @param aControl       control workset
  /// @param aConfig        configuration workset
  /// @param aResult        result workset
  /// @param aScale         scalar 
  virtual 
  void 
  evaluate(
      const Plato::SpatialDomain                         & aSpatialDomain,
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
      const Plato::Scalar                                & aScale
  ) 
  const = 0;

  /// @fn evaluate
  /// @brief pure virtual method: evaluates current density 
  /// @param aState   state workset
  /// @param aControl control workset
  /// @param aConfig  configuration workset
  /// @param aResult  result workset
  /// @param aScale   scalar 
  virtual
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
      const Plato::Scalar                                & aScale
  ) 
  const = 0;
};

}
// namespace Plato