/*
 *  FactorySourceEvaluator_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "elliptic/electrical/SourceEvaluator.hpp"

namespace Plato
{

/// @class FactorySourceEvaluator
/// @brief factory of source evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactorySourceEvaluator
{
public:
  /// @brief class constructor
  FactorySourceEvaluator();
  
  /// @brief class destructor
  ~FactorySourceEvaluator();

  /// @brief create a source evaluator
  /// @param aMaterialName name of input material parameter list in input file
  /// @param aParamList    input problem parameters
  /// @return shared pointer
  std::shared_ptr<Plato::SourceEvaluator<EvaluationType>> 
  create(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList
  );

};

}