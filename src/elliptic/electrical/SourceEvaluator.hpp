/*
 * SourceEvaluator.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "SpatialModel.hpp"

namespace Plato
{

template<typename EvaluationType>
class SourceEvaluator
{
private:
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

public:
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
};
// class SourceEvaluator

}