/*
 * ProblemEvaluatorBase.cpp
 *
 *  Created on: June 21, 2023
 */

#pragma once

#include "Solutions.hpp"
#include "base/Database.hpp"
#include "base/SupportedParamOptions.hpp"

namespace Plato
{

/// @class ProblemEvaluatorBase
/// @brief base class for problem evaluators
class ProblemEvaluatorBase
{
public:
  /// @fn getSolution
  /// @brief get state solution
  /// @return solutions database
  virtual 
  Plato::Solutions
  getSolution() = 0;

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in,out] aDatabase range and domain database
  virtual 
  void 
  updateProblem(
    Plato::Database & aDatabase
  ) = 0;

  /// @fn analyze
  /// @brief analyze physics of interests, solution is saved into the database
  /// @param [in,out] aDatabase range and domain database
  virtual 
  void
  analyze(
    Plato::Database & aDatabase
  ) = 0;

  /// @fn residual
  /// @brief evaluate residual, residual is save into the database
  /// @param [in,out] aDatabase range and domain database
  virtual
  void
  residual(
    Plato::Database & aDatabase
  ) = 0;

  /// @fn criterionValue
  /// @brief evaluate criterion 
  /// @param [in]     aName     criterion name
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar
  virtual
  Plato::Scalar
  criterionValue(
    const std::string     & aName,
          Plato::Database & aDatabase
  ) = 0;

  /// @fn criterionGradient
  /// @brief compute criterion gradient
  /// @param [in]     aEvalType evaluation type, compute gradient with respect to a quantity of interests
  /// @param [in]     aName     criterion name
  /// @param [in,out] aDatabase range and domain database
  /// @return scalar vector
  virtual
  Plato::ScalarVector
  criterionGradient(
    const Plato::evaluation_t & aEvalType,
    const std::string         & aName,
          Plato::Database     & aDatabase
  ) = 0;

  /// @fn criterionIsLinear
  /// @brief return true if criterion is linear; otherwise, return false
  /// @param [in] aName criterion name
  /// @return boolean
  virtual
  bool  
  criterionIsLinear(
    const std::string & aName
  ) = 0;

};

} // namespace Plato
