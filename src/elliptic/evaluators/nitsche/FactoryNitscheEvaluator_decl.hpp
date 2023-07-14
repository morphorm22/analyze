/*
 * FactoryNitscheEvaluator_decl.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "bcs/dirichlet/nitsche/NitscheEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class FactoryNitscheEvaluator
/// @brief creates evaluator used to weakly enforce dirichlet boundary conditions via nitsche's method
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryNitscheEvaluator
{
public:
  /// @fn create
  /// @brief create nitsche integral evaluator
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  /// @return standard shared pointer
  std::shared_ptr<Plato::NitscheEvaluator>
  create(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );

private:
  /// @fn createLinearNitscheEvaluator
  /// @brief create nitsche integral evaluator for linear elliptic problems
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  /// @return standard shared pointer
  std::shared_ptr<Plato::NitscheEvaluator>
  createLinearNitscheEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );

  /// @fn createNonlinearNitscheEvaluator
  /// @brief create nitsche integral evaluator for nonlinear elliptic problems
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  /// @return standard shared pointer
  std::shared_ptr<Plato::NitscheEvaluator>
  createNonlinearNitscheEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );
};

} // namespace Elliptic

} // namespace Plato