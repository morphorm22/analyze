/*
 * VectorFunctionBase.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

namespace Plato
{

namespace Elliptic
{

/// @class VectorFunctionBase
/// @brief vector function base class for elliptic problems. interface between plato problem and residual evaluator
class VectorFunctionBase
{
public:
  /// @brief class destructor
  virtual ~VectorFunctionBase(){}

  /// @fn value
  /// @brief evaluate vector function
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar vector
  virtual
  Plato::ScalarVector
  value(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) = 0;

  /// @fn jacobianState
  /// @brief evaluate jacobian with respect to states
  /// @param [in] aDatabase  function domain and range database
  /// @param [in] aCycle     scalar, e.g.; time step
  /// @param [in] aTranspose apply transpose
  /// @return teuchos reference counter pointer 
  virtual
  Teuchos::RCP<Plato::CrsMatrixType>
  jacobianState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle,
          bool              aTranspose
  ) = 0;

  /// @fn jacobianControl
  /// @brief evaluate jacobian with respect to controls
  /// @param [in] aDatabase  function domain and range database
  /// @param [in] aCycle     scalar, e.g.; time step
  /// @param [in] aTranspose apply transpose
  /// @return teuchos reference counter pointer 
  virtual
  Teuchos::RCP<Plato::CrsMatrixType>
  jacobianControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle,
          bool              aTranspose
  ) = 0;

  /// @fn jacobianConfig
  /// @brief evaluate jacobian with respect to configuration
  /// @param [in] aDatabase  function domain and range database
  /// @param [in] aCycle     scalar, e.g.; time step
  /// @param [in] aTranspose apply transpose
  /// @return teuchos reference counter pointer 
  virtual
  Teuchos::RCP<Plato::CrsMatrixType>
  jacobianConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle,
          bool              aTranspose
  ) = 0;
};

} // namespace Elliptic


} // namespace Plato
