/*
 * FactoryProblemEvaluator_decl.hpp
 *
 *  Created on: June 27, 2023
 */

#pragma once

#include <memory>

#include "SpatialModel.hpp"
#include "solver/ParallelComm.hpp"
#include "base/ProblemEvaluatorBase.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class FactoryProblemEvaluator
/// @brief creates elliptic problem evaluator
/// @tparam PhysicsType defines physics and related physical quantity of interests for this problem
template<typename PhysicsType>
class FactoryProblemEvaluator
{
public:
  /// @brief class constructor
  FactoryProblemEvaluator();
  /// @brief class destructor
  ~FactoryProblemEvaluator(){}

  /// @fn create 
  /// @brief create problem evaluator
  /// @param [in] aParamList    input problem parameters
  /// @param [in] aSpatialModel contains mesh and model information
  /// @param [in] aDataMap      output database
  /// @param [in] aMachine      mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::ProblemEvaluatorBase>
  create(
    Teuchos::ParameterList & aParamList,
    Plato::SpatialModel    & aSpatialModel,
    Plato::DataMap         & aDataMap,
    Plato::Comm::Machine   & aMachine
  );

};

} // namespace Elliptic

} // namespace Plato