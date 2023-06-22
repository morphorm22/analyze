/*
 * Problem_decl.hpp
 *
 *  Created on: June 21, 2023
 */

#pragma once

#include "PlatoAbstractProblem.hpp"

#include "Solutions.hpp"
#include "SpatialModel.hpp"

#include "solver/ParallelComm.hpp"

#include "base/Database.hpp"
#include "base/ProblemEvaluatorBase.hpp"


namespace Plato
{

namespace Elliptic
{

/// @class Problem
/// @brief class for elliptic problems
/// @tparam PhysicsType defines physics and related quantity of interests for this physics 
template<typename PhysicsType>
class Problem : public Plato::AbstractProblem
{
private:
  /// @brief contains mesh and model information
  Plato::SpatialModel mSpatialModel;
  /// @brief save state if true
  bool mSaveState = false;
  /// @brief partial differential equation type
  std::string mTypePDE; 
  /// @brief simulated physics
  std::string mPhysics; 
  /// @brief elliptic problem evaluation interface
  std::shared_ptr<Plato::ProblemEvaluatorBase> mProblemEvaluator; 

public:
  /// @brief class constructor
  /// @param [in] aMesh       mesh interface
  /// @param [in] aParamList input problem parameters
  /// @param [in] aMachine    mpi wrapper
  Problem(
    Plato::Mesh              aMesh,
    Teuchos::ParameterList & aParamList,
    Plato::Comm::Machine     aMachine
  );

  /// @brief class destructor
  ~Problem(){}

  /// @fn getSolution
  /// @brief get state solution
  /// @return solutions database
  Plato::Solutions 
  getSolution() 
  const;

  /// @fn criterionIsLinear
  /// @brief return true if criterion is linear; otherwise, return false
  /// @param [in] aName criterion name
  /// @return boolean
  bool
  criterionIsLinear(
    const std::string & aName
  );

  /// @fn output 
  /// @brief output state solution and requested quantities of interests to visualization file
  /// @param [in] aFilepath output file name 
  void 
  output(
    const std::string & aFilepath
  );

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aControl  control variables
  /// @param [in] aSolution current state solution
  void 
  updateProblem(
    const Plato::ScalarVector & aControls, 
    const Plato::Solutions    & aSolution
  );

  /// @fn solution
  /// @brief solve for state solution
  /// @param [in] aControl control variables
  /// @return state solution database
  Plato::Solutions
  solution(
    const Plato::ScalarVector & aControls
  );

  /// @fn criterionValue
  /// @brief evaluate criterion
  /// @param [in] aControl  control variables
  /// @param [in] aSolution current state solution
  /// @param [in] aName     criterion name
  /// @return scalar
  Plato::Scalar
  criterionValue(
    const Plato::ScalarVector & aControls,
    const Plato::Solutions    & aSolution,
    const std::string         & aName
  );

  /// @fn criterionValue
  /// @brief evaluate criterion
  /// @param [in] aControl control variables
  /// @param [in] aName    criterion name
  /// @return scalar
  Plato::Scalar
  criterionValue(
    const Plato::ScalarVector & aControls,
    const std::string         & aName
  );

  /// @fn criterionGradient
  /// @brief compute criterion gradient with respect to the controls
  /// @param [in] aControl  control variables
  /// @param [in] aSolution current state solution
  /// @param [in] aName     criterion name
  /// @return scalar vector
  Plato::ScalarVector
  criterionGradient(
    const Plato::ScalarVector & aControls,
    const Plato::Solutions    & aSolution,
    const std::string         & aName
  );
  
  /// @fn criterionGradient
  /// @brief compute criterion gradient with respect to the controls
  /// @param [in] aControl  control variables
  /// @param [in] aName     criterion name
  /// @return scalar vector  
  Plato::ScalarVector
  criterionGradient(
    const Plato::ScalarVector & aControls,
    const std::string         & aName
  );

  /// @fn criterionGradientX
  /// @brief compute criterion gradient with respect to the configuration
  /// @param [in] aControl  control variables
  /// @param [in] aSolution current state solution
  /// @param [in] aName     criterion name
  /// @return scalar vector
  Plato::ScalarVector
  criterionGradientX(
      const Plato::ScalarVector & aControls,
      const Plato::Solutions    & aSolution,
      const std::string         & aName
  );

  /// @fn criterionGradientX
  /// @brief compute criterion gradient with respect to the configuration
  /// @param [in] aControl  control variables
  /// @param [in] aName     criterion name
  /// @return scalar vector
  Plato::ScalarVector
  criterionGradientX(
    const Plato::ScalarVector & aControls,
    const std::string         & aName
  );

private:
  /// @fn buildDatabase
  /// @brief build problem range and domain database
  /// @param [in]     aControls control variables
  /// @param [in,out] aDatabase ranga and domain database
  void
  buildDatabase(
    const Plato::ScalarVector & aControls,
          Plato::Database     & aDatabase
  );

  /// @fn parseSaveOutput
  /// @brief set save state flag
  /// @param aParamList input problem parameters
  void
  parseSaveOutput(
    Teuchos::ParameterList & aParamList
  );

};

} // namespace Elliptic

} // namespace Plato
