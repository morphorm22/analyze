/*
 * SupportedParamOptions.hpp
 *
 *  Created on: May 30, 2023
 */

#pragma once

#include <string>
#include <unordered_map>

namespace Plato
{

namespace Elliptic
{
  
namespace mechanical
{

/// @enum residual
/// @brief supported residual enums for mechanical physics
enum struct residual
{
  LINEAR_MECHANICS=0,
  NONLINEAR_MECHANICS=1
};

/// @struct ResidualEnum
/// @brief Interface between input response type and supported mechanical residual 
struct ResidualEnum
{
private:
  /// @brief map from state response type to supported mechanical residual enum
  std::unordered_map<std::string,Plato::Elliptic::mechanical::residual> s2e = 
  {
    {"linear"   ,Plato::Elliptic::mechanical::residual::LINEAR_MECHANICS},
    {"nonlinear",Plato::Elliptic::mechanical::residual::NONLINEAR_MECHANICS}
  };

public:
  /// @brief return supported mechanical residual enum
  /// @param [in] aResponse state response, linear or nonlinear
  /// @return residual enum
  Plato::Elliptic::mechanical::residual 
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

/// @enum property
/// @brief supported mechanical material property enums
enum struct property
{
  /// @brief Supported mechanical material property enums
  YOUNGS_MODULUS=0, 
  POISSON_RATIO=1, 
  MASS_DENSITY=2, 
  LAME_LAMBDA=3, 
  LAME_MU=4,
  TO_ERSATZ_MATERIAL_EXPONENT=5,
  TO_MIN_ERSATZ_MATERIAL_VALUE=6
};

/// @struct PropEnum
/// @brief interface between input mechanical material property string and supported mechanical material property enum
struct PropEnum
{
private:
  /// @brief map from input mechanical material property string to supported mechanical material property enum
  std::unordered_map<std::string,Plato::Elliptic::mechanical::property> s2e = {
    {"youngs modulus"  ,Plato::Elliptic::mechanical::property::YOUNGS_MODULUS},
    {"poissons ratio"  ,Plato::Elliptic::mechanical::property::POISSON_RATIO},
    {"density"         ,Plato::Elliptic::mechanical::property::MASS_DENSITY},
    {"lame lambda"     ,Plato::Elliptic::mechanical::property::LAME_LAMBDA},
    {"lame mu"         ,Plato::Elliptic::mechanical::property::LAME_MU},
    {"penalty exponent",Plato::Elliptic::mechanical::property::TO_ERSATZ_MATERIAL_EXPONENT},
    {"minimum value"   ,Plato::Elliptic::mechanical::property::TO_MIN_ERSATZ_MATERIAL_VALUE}    
  };

public:
  /// @fn get
  /// @brief Return mechanical property enum associated with input string, 
  ///   throw if requested mechanical property is not supported
  /// @param [in] aInput property identifier
  /// @return mechanical property enum
  Plato::Elliptic::mechanical::property 
  get(
    const std::string &aInput
  ) const;

private:
  /// @fn getErrorMsg
  /// @brief return error message
  /// @param [in] aInProperty property name enter in input deck
  /// @return error message string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  ) const;
};

/// @enum material
/// @brief supported mechanical material enums
enum struct material
{
  /// @brief Supported mechanical material enums
  HYPERELASTIC_KIRCHHOFF=0,
  HYPERELASTIC_NEOHOOKEAN=1,
};

/// @struct PropEnum
/// @brief interface between input mechanical material string and supported mechanical material enum
struct MaterialEnum
{
private:
  /// @brief map from input mechanical material string to supported mechanical material enum
  std::unordered_map<std::string,Plato::Elliptic::mechanical::material> s2e = {
    {"hyperelastic kirchhoff"  ,Plato::Elliptic::mechanical::material::HYPERELASTIC_KIRCHHOFF},
    {"hyperelastic neo-hookean",Plato::Elliptic::mechanical::material::HYPERELASTIC_NEOHOOKEAN},
  };

public:
  /// @fn get
  /// @brief Return mechanical material enum associated with input string, 
  ///   throw if requested mechanical material is not supported
  /// @param [in] aInput mechanical material identifier
  /// @return mechanical material enum
  Plato::Elliptic::mechanical::material 
  get(
    const std::string &aInput
  ) const;

private:
  /// @fn getErrorMsg
  /// @brief return error message
  /// @param [in] aInProperty property name enter in input deck
  /// @return error message string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  ) const;
  
};

/// @enum criterion
/// @brief supported criterion enums for nonlinear mechanical physics
enum struct criterion
{
  VOLUME=0,
  KIRCHHOFF_ENERGY_POTENTIAL=1,
  NEO_HOOKEAN_ENERGY_POTENTIAL=2,
};

/// @struct CriterionEnum
/// @brief interface between input string and supported nonlinear mechanical criterion 
struct CriterionEnum
{
private:
  /// @brief map from string to supported mechanical criterion enum
  std::unordered_map<std::string,Plato::Elliptic::mechanical::criterion> s2e = {
    {"volume"                      ,Plato::Elliptic::mechanical::criterion::VOLUME},
    {"kirchhoff energy potential"  ,Plato::Elliptic::mechanical::criterion::KIRCHHOFF_ENERGY_POTENTIAL},
    {"neo-hookean energy potential",Plato::Elliptic::mechanical::criterion::NEO_HOOKEAN_ENERGY_POTENTIAL}
  };

public:
  /// @brief Return mechanical criterion enum associated with input string,
  ///   throw error if requested criterion is not supported
  /// @param [in] aInput string property
  /// @return mechanical criterion enum
  Plato::Elliptic::mechanical::criterion 
  get(const std::string &aInput) 
  const;

private:
  /// @brief Return error message if input option is not supported
  /// @param [in] aInProperty input string - parsed from input file
  /// @return string
  std::string
  getErrorMsg(const std::string & aInProperty)
  const;
};

}
// namespace mechanical

} // namespace Elliptic

}
// namespace Plato