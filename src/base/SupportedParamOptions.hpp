/*
 * SupportedParamOptions.hpp
 *
 *  Created on: June 20, 2023
 */

#pragma once

#include <unordered_map>

namespace Plato
{

/// @brief capability production classification enum
enum struct production_t
{
  SUPPORTED=0,
  EXPERIMENTAL=1,
};

/// @brief partial differential equation classification enum
enum struct pde_t
{
  ELLIPTIC=0,
  HYPERBOLIC=1,
  PARABOLIC=2,
};

/// @brief physics enum
enum struct physics_t
{
  // supported
  THERMAL=0,
  MECHANICAL=1,
  ELECTRICAL=2,
  THERMOMECHANICAL=3,
  HELMHOLTZ_FILTER=4,
  // experimental
  PLASTICITY=5,
  THERMOPLASTICITY=6,
  ELECTROMECHANICAL=7,
  INCOMPRESSIBLE_FLUID=8,
};

/// @brief physics response enum
enum struct response_t
{
  LINEAR=0, 
  NONLINEAR=1
};

/// @brief physics coupling strategies
enum struct coupling_t
{
  STAGGERED=0, 
  MONOLITHIC=1
};

/// @struct PhysicsEnum
/// @brief maps input physics, response, and coupling strings to corresponding supported enums
struct PhysicsEnum
{
private:
  /// @brief map from input physics string to problem physics / production classification enum pair
  std::unordered_map<std::string,std::pair<Plato::physics_t,Plato::production_t>> sp2e = 
  {
    {"thermal"               ,{ Plato::physics_t::THERMAL,Plato::production_t::SUPPORTED} },
    {"mechanical"            ,{ Plato::physics_t::MECHANICAL,Plato::production_t::SUPPORTED} },
    {"electrical"            ,{ Plato::physics_t::ELECTRICAL,Plato::production_t::SUPPORTED} },
    {"plasticity"            ,{ Plato::physics_t::PLASTICITY,Plato::production_t::EXPERIMENTAL} },
    {"thermomechanical"      ,{ Plato::physics_t::THERMOMECHANICAL,Plato::production_t::SUPPORTED} },
    {"helmholtz filter"      ,{ Plato::physics_t::HELMHOLTZ_FILTER,Plato::production_t::SUPPORTED} },
    {"thermoplasticity"      ,{ Plato::physics_t::THERMOPLASTICITY,Plato::production_t::EXPERIMENTAL} },
    {"electromechanical"     ,{ Plato::physics_t::ELECTROMECHANICAL,Plato::production_t::EXPERIMENTAL} },
    {"incompressible fluids" ,{ Plato::physics_t::INCOMPRESSIBLE_FLUID,Plato::production_t::EXPERIMENTAL} }
  };

  /// @brief map from input coupling method string to supported coupling method enum
  std::unordered_map<std::string,Plato::coupling_t> sc2e = 
  {
    {"staggered" ,Plato::coupling_t::STAGGERED},
    {"monolithic",Plato::coupling_t::MONOLITHIC}
  };

  /// @brief map from input physics response string to supported physics response enum
  std::unordered_map<std::string,Plato::response_t> sr2e = 
  {
    {"linear"   ,Plato::response_t::LINEAR},
    {"nonlinear",Plato::response_t::NONLINEAR}
  };

  /// @brief map from input partial differential equation (pde) string to supported pde enum
  std::unordered_map<std::string,Plato::pde_t> se2e = 
  {
    {"elliptic"  ,Plato::pde_t::ELLIPTIC},
    {"parabolic" ,Plato::pde_t::PARABOLIC},
    {"hyperbolic",Plato::pde_t::HYPERBOLIC}
  };

public:
  /// @fn physics
  /// @brief return supported physics enum
  /// @param [in] aPhysics input physics
  /// @return physics enum
  Plato::physics_t 
  physics(
    const std::string & aPhysics
  ) const;

  /// @fn coupling
  /// @brief return supported coupling method enum
  /// @param [in] aCoupling input coupling method
  /// @return coupling method enum
  Plato::coupling_t
  coupling(
    const std::string & aCoupling
  ) const;

  /// @fn response
  /// @brief return supported physics response enum
  /// @param [in] aResponse input physics response
  /// @return physics response enum
  Plato::response_t
  response(
    const std::string & aResponse
  ) const;

  /// @fn pde
  /// @brief return supported partial differential equation (pde) enum
  /// @param [in] aPDE input pde
  /// @return pde enum
  Plato::pde_t
  pde(
    const std::string & aPDE
  ) const;

  /// @fn production
  /// @brief returns true if physics simulation capabilities are considered production 
  /// @param [in] aPhysics input physics 
  /// @return boolean
  bool
  production(
    const std::string & aPhysics
  ) const;

private:
  /// @fn getErrorMsgPhysics
  /// @brief Return error message if physics is not supported
  /// @param [in] aPhysics string - input physics type
  /// @return error message string
  std::string
  getErrorMsgPhysics(
    const std::string & aPhysics
  ) const;

  /// @fn getErrorMsgCoupling
  /// @brief Return error message if physics coupling method is not supported
  /// @param [in] aCoupling string - input physics coupling method
  /// @return error message string
  std::string
  getErrorMsgCoupling(
    const std::string & aCoupling
  ) const;

  /// @fn getErrorMsgResponse
  /// @brief Return error message if physics response is not supported
  /// @param [in] aResponse string - input physics response
  /// @return error message string
  std::string
  getErrorMsgResponse(
    const std::string & aResponse
  ) const;

  /// @fn getErrorMsgPDE
  /// @brief Return error message if partial differential equation (pde) is not supported
  /// @param [in] aPDE string - input pde
  /// @return error message string
  std::string
  getErrorMsgPDE(
    const std::string & aPDE
  ) const;
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
