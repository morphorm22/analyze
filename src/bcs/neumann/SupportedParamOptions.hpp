/*
 * SupportedParamOptions.hpp
 *
 *  Created on: July 5, 2023
 */

#pragma once

namespace Plato
{

/// @brief supported neumann boundary conditions
enum struct neumann_bc
{
  UNIFORM = 1,
  UNIFORM_PRESSURE = 2,
  UNIFORM_COMPONENT = 3,
  FOLLOWER_PRESSURE = 4,
};

/// @struct NeumannEnum
/// @brief maps input neumann boundary condition string to corresponding supported enum
struct NeumannEnum
{
public:
  /// @brief map from supported neumann boundary condition string to corresponding enum
  std::unordered_map<std::string,Plato::neumann_bc> s2e = 
  {
    {"uniform"          ,Plato::neumann_bc::UNIFORM},
    {"uniform pressure" ,Plato::neumann_bc::UNIFORM_PRESSURE},
    {"uniform component",Plato::neumann_bc::UNIFORM_COMPONENT},
    {"follower pressure",Plato::neumann_bc::FOLLOWER_PRESSURE},
  };

public:
  /// @brief map neumann boundary condition string to corresponding supported enum
  /// @param [in] aNBC string
  /// @return supported enum
  Plato::neumann_bc 
  bc(
    const std::string & aNBC
  ) const;

private:
  /// @fn getErrorMsg
  /// @brief Return error message if neumann boundary condition is not supported
  /// @param [in] aNBC string - input neumann boundary condition type
  /// @return error message string
  std::string
  getErrorMsg(
    const std::string & aNBC
  ) const;
};

} // namespace Plato