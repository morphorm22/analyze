/*
 * SupportedParamOptions.hpp
 *
 *  Created on: June 20, 2023
 */

#pragma once

namespace Plato
{

/// @brief supported evaluation types
enum struct evaluation_t
{
  /// @brief criterion value evaluation type
  VALUE=0, 
  /// @brief criterion gradient with respect to the states evaluation type
  GRAD_U=1, 
  /// @brief criterion gradient with respect to the controls evaluation type
  GRAD_Z=2, 
  /// @brief criterion gradient with respect to the configuration evaluation type
  GRAD_X=3,
  /// @brief criterion gradient with respect to the node states evaluation type
  GRAD_N=4
};

} // namespace Plato
