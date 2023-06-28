/*
 * SupportedParamOptions.hpp
 *
 *  Created on: June 20, 2023
 */

#pragma once

#include <unordered_map>

namespace Plato
{

enum struct physics_t
{
  THERMAL=0,
  MECHANICAL=1,
  ELECTRICAL=2,
  THERMOMECHANICAL=3,
};

enum struct coupling_t
{
  STAGGERED=0,
  MONOLITHIC=1
};

struct PhysicsEnum
{
private:
  /// @brief map from state response type to supported mechanical residual enum
  std::unordered_map<std::string,Plato::physics_t> sp2e = 
  {
    {"thermal"         ,Plato::physics_t::THERMAL},
    {"mechanical"      ,Plato::physics_t::MECHANICAL},
    {"electrical"      ,Plato::physics_t::ELECTRICAL},
    {"thermomechanical",Plato::physics_t::THERMOMECHANICAL}
  };

  std::unordered_map<std::string,Plato::coupling_t> sc2e = 
  {
    {"staggered" ,Plato::coupling_t::STAGGERED},
    {"monolithic",Plato::coupling_t::MONOLITHIC}
  };

public:
  /// @fn get
  /// @brief return supported physics enum
  /// @param [in] aPhysics input physics
  /// @return physics enum
  Plato::physics_t 
  physics(
    const std::string & aPhysics
  ) 
  const;

  Plato::coupling_t
  coupling(
    const std::string & aCoupling
  )
  const;

private:
  /// @fn getErrorMsgPhysics
  /// @brief Return error message if physics is not supported
  /// @param [in] aPhysics string - physics type
  /// @return error message string
  std::string
  getErrorMsgPhysics(
    const std::string & aPhysics
  )
  const;

  /// @fn getErrorMsgCoupling
  /// @brief Return error message if physics coupling method is not supported
  /// @param [in] aPhysics string - physics coupling method
  /// @return error message string
  std::string
  getErrorMsgCoupling(
    const std::string & aPhysics
  )
  const;
};

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
