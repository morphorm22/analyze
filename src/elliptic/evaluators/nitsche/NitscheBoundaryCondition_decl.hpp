/*
 * NitscheBoundaryCondition_decl.hpp
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

/// NitscheBoundaryCondition
/// @brief weak enforcement of dirichlet boundary conditions via nitsche's method
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class NitscheBoundaryCondition : public Plato::NitscheEvaluator
{
private:
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief nitsche's integral evaluator
  std::shared_ptr<BaseClassType> mNitscheBC;

public:
  /// @brief class constructor
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  NitscheBoundaryCondition(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );

  /// @fn evaluate
  /// @brief enforces weak dirichlet boundary conditions via nitsche's method
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