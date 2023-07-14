/*
 * NitscheLinearMechanics_decl.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"

#include "bcs/dirichlet/nitsche/NitscheEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class NitscheLinearMechanics
/// @brief weak enforcement of the dirichlet boundary conditions in linear elastostatic problems using nitsche's method
///
/// \f[
///   -\int_{\Gamma_D}\delta{u}_i\left(\sigma_{ij} n_j}\right)d\Gamma
///   +\int_{\Gamma_D}\delta\left(\sigma_{ij} n_j\right)\left(u_i-u_i^{D}\right)d\Gamma
///   +\int_{\Gamma_D}\gamma_{N}^{u}\delta{u}_i\left(u_i-u_i^{D}\right)d\Gamma
/// \f]
///
/// where \f$u_i^D\f$ is the dirichlet displacement enforced on dirichlet boundary \f$\Gamma_D\f$, 
/// \f$\gamma_{N}^{u}\f$ is a penalty parameter chosen to achieve a desired accuracy in satisfying the dirichlet
/// boundary conditions, \f$u_i\f$ is the displacement field, \f$n_j\f$ is the normal vector to the surface where
/// dirichlet boundary conditions are applied, and \f$\sigma_{ij}\f$ is the second order stress tensor. A 
/// non-symmetric nitsche formulation is considered in this work, see for example Burman (2012) and Schillinger 
/// et al. (2016a).
///
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class NitscheLinearMechanics : public Plato::NitscheEvaluator
{
private:
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief list of nitsche boundary condition evaluators 
  std::vector<std::shared_ptr<Plato::NitscheEvaluator>> mEvaluators;

public:
  /// @brief class constructor
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  NitscheLinearMechanics(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );

  /// @fn evaluate
  /// @brief evaluate nitsche's integral for all side set cells
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        scalar
  /// @param [in]     aScale        scalar
  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  );
};

} // namespace Elliptic

} // namespace Plato