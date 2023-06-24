/*
 * SupportedEllipticProblemOptions.hpp
 *
 *  Created on: June 22, 2023
 */

#pragma once

namespace Plato
{

namespace Elliptic
{

enum struct residual_t
{
  LINEAR_THERMAL = 0,
  LINEAR_MECHANICAL = 1,
  LINEAR_ELECTRICAL = 2,
  LINEAR_THERMO_MECHANICAL = 3,
  LINEAR_ELECTRO_MECHANICAL = 4,
  NONLINEAR_MECHANICAL = 5,
  NONLINEAR_THERMO_MECHANICAL = 6
};

struct ResidualEnum
{
private:
  /// @brief map from input physics to supported elliptic residual enum type
  std::unordered_map<std::string,Plato::Elliptic::residual_t> s2e = {
    {"thermal"                   ,Plato::Elliptic::residual_t::LINEAR_THERMAL},
    {"mechanical"                ,Plato::Elliptic::residual_t::LINEAR_MECHANICAL},
    {"electrical"                ,Plato::Elliptic::residual_t::LINEAR_ELECTRICAL},
    {"thermomechanical"          ,Plato::Elliptic::residual_t::LINEAR_THERMO_MECHANICAL},
    {"electromechanical"         ,Plato::Elliptic::residual_t::LINEAR_ELECTRO_MECHANICAL},
    {"nonlinear mechanical"      ,Plato::Elliptic::residual_t::NONLINEAR_MECHANICAL},
    {"nonlinear thermomechanical",Plato::Elliptic::residual_t::NONLINEAR_THERMO_MECHANICAL}
  };

public:
  Plato::Elliptic::residual_t 
  get(
    const std::string &aInput
  ) const;

private:
  std::string
  getErrorMsg(
    const std::string & aInProperty
  ) const;

};

}

}