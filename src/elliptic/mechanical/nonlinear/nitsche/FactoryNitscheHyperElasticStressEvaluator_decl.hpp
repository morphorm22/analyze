/*
 *  FactoryNitscheHyperElasticStressEvaluator_decl.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include <memory>

#include "bcs/dirichlet/nitsche/BoundaryFluxEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class FactoryNitscheHyperElasticStressEvaluator
/// @brief creates stress evaluator use to weakly enforce dirichlet boundary conditions 
/// in nonlinear mechanical problems via nitsche's method
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryNitscheHyperElasticStressEvaluator
{
public:
  /// @fn createTrialEvaluator
  /// @brief create trial stress evaluator
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  /// @return standard shared pointer
  std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> 
  createTrialEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );

  /// @fn createTestEvaluator
  /// @brief create test stress evaluator
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  /// @return standard shared pointer
  std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> 
  createTestEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );

  /// @fn getMaterialName
  /// @brief return material name
  /// @param [in] aParamList input problem parameters
  /// @return string
  std::string 
  getMaterialName(
    const Teuchos::ParameterList & aParamList
  );

};

} // namespace Elliptic

} // namespace Plato
