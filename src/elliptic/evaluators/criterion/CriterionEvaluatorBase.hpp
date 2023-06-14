#pragma once

#include "base/Database.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class CriterionEvaluatorBase
/// @brief criterion evaluator base class
class CriterionEvaluatorBase
{
public:
  /// @brief class destructor
  virtual ~CriterionEvaluatorBase(){}

  /// @fn name
  /// @brief return criterion evaluator name
  /// @return string
  virtual 
  std::string 
  name() 
  const = 0;

  /// @fn isLinear
  /// @brief return true if scalar function is linear
  /// @return boolean
  virtual
  bool 
  isLinear() 
  const = 0;

  /// @fn value
  /// @brief evaluate criterion
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar
  virtual 
  Plato::Scalar
  value(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const = 0;

  /// @fn gradientControl
  /// @brief compute partial derivative with respect to the controls
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  virtual 
  Plato::ScalarVector
  gradientControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const = 0;

  /// @fn gradientState
  /// @brief compute partial derivative with respect to the states
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  virtual 
  Plato::ScalarVector
  gradientState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const = 0;

  /// @fn gradientConfig
  /// @brief compute partial derivative with respect to the configuration
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  virtual 
  Plato::ScalarVector
  gradientConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const = 0;

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  virtual 
  void 
  updateProblem(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const = 0;

}; // class CriterionEvaluatorBase

/// @brief evaluation type
enum struct evaluator_t
{
  VALUE=0, 
  GRAD_U=1, 
  GRAD_Z=2, 
  GRAD_X=3,
};

} // namespace Elliptic

} // namespace Plato
