/*
 * ScalarFunctionBase.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

#include "base/Database.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class ScalarFunctionBase
/// @brief scalar function base class for elliptic problems. interface between plato problem and criterion evaluator
class ScalarFunctionBase
{
public:
  /// @brief class destructor
  virtual ~ScalarFunctionBase(){}

  /// @fn name
  /// @brief return scalar function name
  /// @return string
  virtual std::string name() const = 0;

  /// @fn isLinear
  /// @brief return true if scalar function is linear
  /// @return boolean
  virtual bool isLinear() const = 0;
  
  /// @fn value
  /// @brief evaluate scalar function
  /// @param aDatabase output database
  /// @param aCycle    scalar, e.g.; time step
  /// @return scalar
  virtual Plato::Scalar
  value(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) = 0;
  
  /// @fn gradientControl
  /// @brief compute the partial derivative of the scalar function with respect to controls
  /// @param aDatabase output database
  /// @param aCycle    scalar, e.g.; time step
  /// @return scalar plato vector
  virtual Plato::ScalarVector
  gradientControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) = 0;
  
  /// @fn gradientState
  /// @brief compute the partial derivative of the scalar function with respect to states
  /// @param aDatabase output database
  /// @param aCycle    scalar, e.g.; time step
  /// @return scalar plato vector
  virtual Plato::ScalarVector
  gradientState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) = 0;

  /// @fn gradientConfig
  /// @brief compute the partial derivative of the scalar function with respect to configuration
  /// @param aDatabase output database
  /// @param aCycle    scalar, e.g.; time step
  /// @return scalar plato vector
  virtual Plato::ScalarVector
  gradientConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) = 0;
};

/// @brief evaluation type
enum struct evaluator_t
{
  VALUE=0, 
  GRAD_U=1, 
  GRAD_Z=2, 
  GRAD_X=3,
};

} // namespace Elliptic


} // namespace Plato
