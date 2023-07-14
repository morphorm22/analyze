/*
 * NitscheLinearThermoStatics_def.hpp
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

/// @class NitscheLinearThermoStatics
/// @brief weak enforcement of the dirichlet boundary conditions in linear thermostatic problems using nitsche's method
///
/// \f[
///   -\int_{\Gamma_D}\delta\theta\left(q_i n_i}\right)d\Gamma
///   +\int_{\Gamma_D}\delta\left(q_i n_i\right)\left(\theta-\theta^{D}\right)d\Gamma
///   +\int_{\Gamma_D}\gamma_{N}^{\theta}\delta\theta\left(\theta-\theta^{D}\right)d\Gamma
/// \f]
///
/// where \f$\theta^D\f$ is the dirichlet temperature field enforced on dirichlet boundary \f$\Gamma_D\f$, 
/// \f$\gamma_{N}^{\theta}\f$ is a penalty parameter chosen to achieve a desired accuracy in satisfying the Dirichlet
/// boundary conditions, \f$\theta\f$ is the temperature field, \f$n_i\f$ is the normal vector to the surface where
/// dirichlet boundary conditions are applied, and \f$q_i\f$ is the heat flux. A non-symmetric nitsche formulation is
/// considered in this work, see for example Burman (2012) and Schillinger et al. (2016a).
///
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class NitscheLinearThermoStatics : public Plato::NitscheEvaluator
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
  NitscheLinearThermoStatics(
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
