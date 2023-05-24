/*
 * FactoryCurrentDensityEvaluator_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include <memory>

#include "elliptic/electrical/CurrentDensityEvaluator.hpp"

namespace Plato
{

/// @class FactoryCurrentDensityEvaluator
/// @brief factory fpr current density evaluators 
/// @tparam EvaluationType 
template<typename EvaluationType>
class FactoryCurrentDensityEvaluator
{
public:
  /// @brief class constructor
  FactoryCurrentDensityEvaluator();

  /// @brief class destructor
  ~FactoryCurrentDensityEvaluator();

  /// @fn create
  /// @brief creates a shared pointed to a current density evaluator
  /// @param aMaterialName name of input material parameter list
  /// @param aFunctionName name of input current density evaluator parameter list
  /// @param aParamList    input problem parameters
  /// @return shared pointer
  std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> 
  create(
    const std::string            & aMaterialName,
    const std::string            & aFunctionName,
          Teuchos::ParameterList & aParamList
  );

};
// class FactoryCurrentDensityEvaluator

}