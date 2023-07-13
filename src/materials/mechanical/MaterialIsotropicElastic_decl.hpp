/*
 * MaterialIsotropicElastic_decl.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "materials/MaterialModel.hpp"

namespace Plato
{

/// @class MaterialIsotropicElastic
/// 
/// @brief material constitutive model for isotropic elastic materials:
///
///  \f[
///    C_{ijkl}=\lambda\delta_{ij}\delta_{kl} + \mu\left(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}\right)
///  \f]
///
/// where \f$\lambda\f$ and \f$\mu\f$ are the Lame constants, \f$\delta\f$ is the Kronecker delta, and 
/// \f$C_{ijkl}\f$ is the fourth-order isotropic material tensor.  
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialIsotropicElastic : public MaterialModel<EvaluationType>
{
public:
  /// @brief class constructor
  MaterialIsotropicElastic(){}

  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  MaterialIsotropicElastic(
    const Teuchos::ParameterList& aParamList
  );

  /// @brief class destructor
  ~MaterialIsotropicElastic(){}

  /// @fn mu
  /// @brief return value of lame constant mu
  /// @return scalar
  Plato::Scalar 
  mu();
  
  /// @fn mu
  /// @brief set value for lame constant mu
  /// @param [in] aValue scalar 
  void 
  mu(
    const Plato::Scalar & aValue
  );

  /// @fn lambda
  /// @brief return value of lame constant lambda
  /// @return scalar
  Plato::Scalar 
  lambda();
  
  /// @fn lambda
  /// @brief set value of lame constant lambda
  /// @param [in] aValue scalar 
  void 
  lambda(
    const Plato::Scalar & aValue
  );

private:
  /// @fn parse
  /// @brief parse input material parameters
  /// @param [in] aParamList input problem parameters 
  void 
  parse(
    const Teuchos::ParameterList& aParamList
  );

  /// @fn computeLameConstants
  /// @brief compute lame constants lambda and mu from input material constants 
  /// E (Young's modulus) and nu (Poisson's ratio) 
  void 
  computeLameConstants();

};

} // namespace Plato