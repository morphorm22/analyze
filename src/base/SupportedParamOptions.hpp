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
  VALUE=0, 
  GRAD_U=1, 
  GRAD_Z=2, 
  GRAD_X=3,
};

} // namespace Plato
