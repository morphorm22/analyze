/*
 * SupportedParamOptions.hpp
 *
 *  Created on: June 27, 2023
 */

#pragma once

#include <string>
#include <unordered_map>

namespace Plato
{

namespace Elliptic
{

namespace thermomechanical
{

/// @enum residual
/// @brief supported residual enums for thermomechanical physics
enum struct residual
{
  LINEAR_THERMO_MECHANICS=0,
  NONLINEAR_THERMO_MECHANICS=1
};

/// @struct ResidualEnum
/// @brief Interface between input state response type input and supported thermomechanical residual 
struct ResidualEnum
{
private:
  /// @brief map from state response type to supported thermomechanical residual enum
  std::unordered_map<std::string,Plato::Elliptic::thermomechanical::residual> s2e = 
  {
    {"linear"   ,Plato::Elliptic::thermomechanical::residual::LINEAR_THERMO_MECHANICS},
    {"nonlinear",Plato::Elliptic::thermomechanical::residual::NONLINEAR_THERMO_MECHANICS}
  };

public:
  /// @brief return supported  elliptic thermomechanical residual enum
  /// @param [in] aResponse state response, linear or nonlinear
  /// @return residual enum
  Plato::Elliptic::thermomechanical::residual 
  get(
    const std::string & aResponse
  ) 
  const;

private:
  /// @fn getErrorMsg
  /// @brief Return error message if response is not supported
  /// @param [in] aResponse string - response type, linear or nonlinear
  /// @return error message string
  std::string
  getErrorMsg(
    const std::string & aResponse
  )
  const;
};

} // namespace thermomechanical

} // namespace Elliptic

} // namespace Plato
