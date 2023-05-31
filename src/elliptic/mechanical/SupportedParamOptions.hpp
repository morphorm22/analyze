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

namespace mechanical
{

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
  std::unordered_map<std::string,Plato::mechanical::property> s2e = {
      {"youngs modulus"  ,Plato::mechanical::property::YOUNGS_MODULUS},
      {"poissons ratio"  ,Plato::mechanical::property::POISSON_RATIO},
      {"density"         ,Plato::mechanical::property::MASS_DENSITY},
      {"lame lambda"     ,Plato::mechanical::property::LAME_LAMBDA},
      {"lame mu"         ,Plato::mechanical::property::LAME_MU},
      {"penalty exponent",Plato::mechanical::property::TO_ERSATZ_MATERIAL_EXPONENT},
      {"minimum value"   ,Plato::mechanical::property::TO_MIN_ERSATZ_MATERIAL_VALUE}    
  };

public:
  /// @fn get
  /// @brief Return mechanical property enum associated with input string, 
  ///   throw if requested mechanical property is not supported
  /// @param [in] aInput property identifier
  /// @return mechanical property enum
  Plato::mechanical::property 
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
  KIRCHHOFF=0,
  NEO_HOOKEAN=1,
};

/// @struct PropEnum
/// @brief interface between input mechanical material string and supported mechanical material enum
struct MaterialEnum
{
private:
  /// @brief map from input mechanical material string to supported mechanical material enum
  std::unordered_map<std::string,Plato::mechanical::material> s2e = {
    {"kirchhoff"  ,Plato::mechanical::material::KIRCHHOFF},
    {"neo-hookean",Plato::mechanical::material::NEO_HOOKEAN},
  };

public:
  /// @fn get
  /// @brief Return mechanical material enum associated with input string, 
  ///   throw if requested mechanical material is not supported
  /// @param [in] aInput mechanical material identifier
  /// @return mechanical material enum
  Plato::mechanical::material 
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

}
// namespace mechanical

}
// namespace Plato