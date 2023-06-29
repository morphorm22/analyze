/*
 * SupportedParamOptions.hpp
 *
 *  Created on: May 23, 2023
 */

#pragma once

// c++ includes
#include <string>
#include <unordered_map>

namespace Plato
{

namespace Elliptic
{
  
/// @namespace electrical
/// @brief electrical physics namespace
namespace electrical 
{

/// @enum residual
/// @brief supported residual enums for electrical physics
enum struct residual
{
  STEADY_STATE_CURRENT=0
};

/// @struct ResidualEnum
/// @brief Interface between input string and supported electrical residual 
struct ResidualEnum
{
private:
  /// @brief map from string to supported electrical residual enum
  std::unordered_map<std::string,Plato::Elliptic::electrical::residual> s2e = {
      {"elliptic",Plato::Elliptic::electrical::residual::STEADY_STATE_CURRENT}
  };

public:
  /// @brief Return electrical residual enum associated with input string,
  ///   throw error if requested residual is not supported
  /// @param [in] aInput string residual type
  /// @return electrical residual enum
  Plato::Elliptic::electrical::residual 
  get(
    const std::string &aInput
  ) 
  const;

private:
  /// @brief Return error message if input option is not supported
  /// @param [in] aInProperty input string - parsed from input file
  /// @return string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  )
  const;
};



/// @enum criterion
/// @brief supported criterion enums for electrical physics
enum struct criterion
{
  TWO_PHASE_POWER_SURFACE_DENSITY=0,
  TWO_PHASE_VOLUME=1
};

/// @struct CriterionEnum
/// @brief Interface between input string and supported electrical criterion 
struct CriterionEnum
{
private:
  /// @brief map from string to supported electrical criterion enum
  std::unordered_map<std::string,Plato::Elliptic::electrical::criterion> s2e = {
      {"two-phase power surface density",Plato::Elliptic::electrical::criterion::TWO_PHASE_POWER_SURFACE_DENSITY},
      {"two-phase volume"               ,Plato::Elliptic::electrical::criterion::TWO_PHASE_VOLUME}
  };

public:
  /// @brief Return electrical criterion enum associated with input string,
  ///   throw error if requested criterion is not supported
  /// @param [in] aInput string property
  /// @return electrical criterion enum
  Plato::Elliptic::electrical::criterion 
  get(
    const std::string &aInput
  ) 
  const;

private:
  /// @brief Return error message if input option is not supported
  /// @param [in] aInProperty input string - parsed from input file
  /// @return string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  )
  const;
};



/// @enum material
/// @brief supported electrical material enums
enum struct material
{
  DIELECTRIC = 0,
  CONDUCTIVE = 1,
  TWO_PHASE_CONDUCTIVE = 2,
};

/// @struct MaterialEnum
/// @brief Interface between input string and supported electrical material constitutive models 
struct MaterialEnum
{
private:
  /// @brief map from string to supported electrical material constitutive model enum
  std::unordered_map<std::string,Plato::Elliptic::electrical::material> s2e = {
      {"two phase conductive",Plato::Elliptic::electrical::material::TWO_PHASE_CONDUCTIVE},
      {"conductive"          ,Plato::Elliptic::electrical::material::CONDUCTIVE},
      {"dielectric"          ,Plato::Elliptic::electrical::material::DIELECTRIC}
  };

public:
  /// @brief Return electrical material constitutive model enum associated with input string,
  ///   throw error if requested material material constitutive model is not supported
  /// @param [in] aInput input material constitutive model string
  /// @return electrical material constitutive model enum
  Plato::Elliptic::electrical::material 
  get(
    const std::string &aInput
  ) 
  const;

private:
  /// @brief Return error message if input option is not supported
  /// @param [in] aInProperty input string - parsed from input file
  /// @return string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  )
  const;
};



/// @enum property
/// @brief supported electrical property enums
enum struct property
{
  /// @brief Supported material property enums for electrical materials
  ELECTRICAL_CONDUCTIVITY=0, 
  OUT_OF_PLANE_THICKNESS=1, 
  MATERIAL_NAME=2, 
  ELECTRICAL_CONSTANT=3, 
  ELECTRICAL_RESISTANCE=4,
  RELATIVE_STATIC_PERMITTIVITY=5,
  TO_ERSATZ_MATERIAL_EXPONENT=6,
  TO_MIN_ERSATZ_MATERIAL_VALUE=7
};

/// @struct PropEnum
/// @brief Interface between input string and supported electrical material properties 
struct PropEnum
{
private:
  /// @brief map from string to supported electrical property enum
  std::unordered_map<std::string,Plato::Elliptic::electrical::property> s2e = {
    {"electrical conductivity"     ,Plato::Elliptic::electrical::property::ELECTRICAL_CONDUCTIVITY},
    {"out-of-plane thickness"      ,Plato::Elliptic::electrical::property::OUT_OF_PLANE_THICKNESS},
    {"material name"               ,Plato::Elliptic::electrical::property::MATERIAL_NAME},
    {"electrical constant"         ,Plato::Elliptic::electrical::property::ELECTRICAL_CONSTANT},
    {"electrical resistance"       ,Plato::Elliptic::electrical::property::ELECTRICAL_RESISTANCE},
    {"relative static permittivity",Plato::Elliptic::electrical::property::RELATIVE_STATIC_PERMITTIVITY},
    {"penalty exponent"            ,Plato::Elliptic::electrical::property::TO_ERSATZ_MATERIAL_EXPONENT},
    {"minimum value"               ,Plato::Elliptic::electrical::property::TO_MIN_ERSATZ_MATERIAL_VALUE}    
  };

public:
  /// @brief Return electrical property enum associated with input string,
  ///   throw error if requested material property is not supported
  /// @param [in] aInput string property
  /// @return electrical property enum
  Plato::Elliptic::electrical::property 
  get(
    const std::string &aInput
  ) 
  const;

private:
  /// @brief Return error message if input option is not supported
  /// @param [in] aInProperty input string - parsed from input file
  /// @return string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  )
  const;
};



/// @enum source_evaluator
/// @brief supported source evaluator enums
enum struct source_evaluator
{
  WEIGHTED_SUM=0,
};

/// @struct SourceEvaluatorEnum
/// @brief interface between input string and supported electrical source evaluators
struct SourceEvaluatorEnum
{
private:
  /// @brief map from input string to supported electrical source evaluator
  std::unordered_map<std::string,Plato::Elliptic::electrical::source_evaluator> s2e = {
    {"weighted sum",Plato::Elliptic::electrical::source_evaluator::WEIGHTED_SUM}
  };

public:
  /// @brief Return source evaluator enum associated with input string,
  ///   throw error if requested source evaluator is not supported
  /// @param [in] aInput string property
  /// @return source evaluator enum
  Plato::Elliptic::electrical::source_evaluator 
  get(
    const std::string &aInput
  ) 
  const;

private:
  /// @brief Return error message if input option is not supported
  /// @param [in] aInProperty input string - parsed from input file
  /// @return string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  )
  const;
};



/// @enum current_density_evaluator
/// @brief supported current density evaluator enums
enum struct current_density_evaluator
{
  TWO_PHASE_DARK_CURRENT_DENSITY=0, 
  TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY=1, 
};

/// @struct CurrentDensitySourceEvaluatorEnum
/// @brief interface between input string and supported current density evaluators
struct CurrentDensitySourceEvaluatorEnum
{
private:
  /// @brief map from input string to supported current density evaluator
  std::unordered_map<std::string,Plato::Elliptic::electrical::current_density_evaluator> s2e = {
    {"two phase dark current density"           ,
      Plato::Elliptic::electrical::current_density_evaluator::TWO_PHASE_DARK_CURRENT_DENSITY
    },
    {"two phase light-generated current density",
      Plato::Elliptic::electrical::current_density_evaluator::TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY
    }
  };

public:
  /// @brief Return current density evaluator enum associated with input string,
  ///   throw error if requested current density evaluator is not supported
  /// @param [in] aInput input string
  /// @return supported enum
  Plato::Elliptic::electrical::current_density_evaluator 
  get(
    const std::string &aInput
  ) 
  const;

private:
  /// @brief Return error message if input option is not supported
  /// @param [in] aInProperty input string - parsed from input file
  /// @return string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  )
  const;
};



/// @enum response
/// @brief supported physics response enums
enum struct response
{
  LINEAR=0, 
  NONLINEAR=1
};

/// @enum current_density_model
/// @brief supported current density models
enum struct current_density_model
{
  /// @brief Supported enums for current density source evaluators
  QUADRATIC=0, 
  CONSTANT=1, 
};

/// @enum current_density
/// @brief supported current density type enums
enum struct current_density
{
  DARK=0, 
  LIGHT=1, 
  SHUNT=2,
};

/// @struct CurrentDensityEnum
/// @brief interface between supported current density evaluator and supported current density models
struct CurrentDensityEnum
{
private:
  /// @brief map from supported current density evaluator enum and supported current density model
  std::unordered_map<
    Plato::Elliptic::electrical::current_density_evaluator,
    std::unordered_map<
      std::string,std::pair<Plato::Elliptic::electrical::current_density,Plato::Elliptic::electrical::response>>> 
  s2e = 
  {
    {
      Plato::Elliptic::electrical::current_density_evaluator::TWO_PHASE_DARK_CURRENT_DENSITY, 
      { 
        { 
          "quadratic",
          {Plato::Elliptic::electrical::current_density::DARK,Plato::Elliptic::electrical::response::NONLINEAR}
        }
      } 
    },
    {
      Plato::Elliptic::electrical::current_density_evaluator::TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY, 
      { 
        { 
          "constant",
          {Plato::Elliptic::electrical::current_density::LIGHT,Plato::Elliptic::electrical::response::LINEAR} 
        }
      } 
    }
  };

  /// @brief interface between input string and supported current density evaluators
  Plato::Elliptic::electrical::CurrentDensitySourceEvaluatorEnum mSourceTermEnums;

public:
  /// @brief Return current density model enum associated with input function and model string,
  ///   throw error if requested current density model is not supported
  /// @param [in] aFunction current density evaluator string
  /// @param [in] aModel    current density model string
  /// @return enum
  Plato::Elliptic::electrical::current_density 
  current_density(
    const std::string & aFunction,
    const std::string & aModel
  ) const;

  /// @brief Return physics response enum associated with input function and model string,
  ///   throw error if requested physics response is not supported
  /// @param [in] aFunction current density evaluator string
  /// @param [in] aModel    current density model string
  /// @return enum
  Plato::Elliptic::electrical::response
  response(
    const std::string & aFunction,
    const std::string & aModel
  ) const;

private:
  /// @brief Return error message if input option is not supported
  /// @param [in] aFunction current density evaluator string
  /// @param [in] aModel    current density model string
  /// @return string
  std::string
  getErrorMsg(
    const std::string & aFunction,
    const std::string & aModel
  ) const;
};

}
// namespace electrical 

} // namespace Elliptic

}
// namespace Plato