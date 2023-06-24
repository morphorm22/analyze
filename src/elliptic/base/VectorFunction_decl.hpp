/*
 * VectorFunction_decl.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

#include <string>
#include <memory>
#include <unordered_map>

#include "Solutions.hpp"
#include "SpatialModel.hpp"
#include "base/Database.hpp"
#include "base/WorksetBase.hpp"
#include "base/ResidualBase.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/base/VectorFunctionBase.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class VectorFunction
/// @brief interface for the evaluation of vector functions in elliptic problems
/// @tparam PhysicsType defines physics and related quantity of interests for this physics 
template<typename PhysicsType>
class VectorFunction : public VectorFunctionBase
{
private:
  /// @brief topological element type
  using ElementType = typename PhysicsType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims          = ElementType::mNumSpatialDims;
  /// @brief number of nodes per element
  static constexpr auto mNumNodesPerCell         = ElementType::mNumNodesPerCell;
  /// @brief number of nodes per element face
  static constexpr auto mNumNodesPerFace         = ElementType::mNumNodesPerFace;
  /// @brief number of state degrees of freedom per node
  static constexpr auto mNumStateDofsPerNode     = ElementType::mNumDofsPerNode;
  /// @brief number of state degrees of freedom per cell
  static constexpr auto mNumStateDofsPerCell     = ElementType::mNumDofsPerCell;
  /// @brief number of node state degrees of freedom per node
  static constexpr auto mNumNodeStateDofsPerNode = ElementType::mNumNodeStatePerNode;
  /// @brief number of node state degrees of freedom per cell
  static constexpr auto mNumNodeStateDofsPerCell = ElementType::mNumNodeStatePerCell;
  /// @brief number of control degree of freedoms per node
  static constexpr auto mNumControlDofsPerNode   = ElementType::mNumControl;
  /// @brief number of configuration degress of freedom per element
  static constexpr auto mNumConfigDofsPerCell    = mNumSpatialDims * mNumNodesPerCell;
  /// @brief scalar types for a given evaluation type
  using ResidualEvalType  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using JacobianUEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
  using JacobianXEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
  using JacobianZEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
  using JacobianNEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientN;
  /// @brief domain (element block) to residual map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mResiduals;
  /// @brief domain (element block) to jacobian with respect to states map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mJacobiansU;
  /// @brief domain (element block) to jacobian with respect to configuration map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mJacobiansX;
  /// @brief domain (element block) to jacobian with respect to controls map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mJacobiansZ;
  /// @brief domain (element block) to jacobian with respect to node states map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mJacobiansN;
  /// @brief output database
  Plato::DataMap & mDataMap;
  /// @brief contains mesh and model information
  const Plato::SpatialModel & mSpatialModel;
  /// @brief interface to workset constructors 
  Plato::WorksetBase<ElementType> mWorksetFuncs;

public:
  /// @brief class constructor
  /// @param [in] aType         partial differential equation type
  /// @param [in] aSpatialModel contains mesh and model information
  /// @param [in] aDataMap      output database
  /// @param [in] aProbParams   input problem parameters
  VectorFunction(
    const std::string            & aType,
    const Plato::SpatialModel    & aSpatialModel,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProbParams
  );

  /// @brief class destructor
  ~VectorFunction(){}

  /// @fn type
  /// @brief get vector function type, which is set by the residual evaluator type
  /// @return residual_t enum
  Plato::Elliptic::residual_t
  type() 
  const;

  /// @fn numDofs
  /// @brief return number of degress of freedom per node
  /// @return integer 
  Plato::OrdinalType 
  numDofs() 
  const;

  /// @fn numNodes
  /// @brief return local number nodes
  /// @return integer 
  Plato::OrdinalType 
  numNodes() 
  const;

  /// @fn numCells
  /// @brief return local number of cells/elements
  /// @return integer
  Plato::OrdinalType 
  numCells() 
  const;

  /// @fn numDofsPerCell
  /// @brief return number of state degree of freedom per cell
  /// @return integer
  Plato::OrdinalType 
  numDofsPerCell() 
  const;

  /// @fn numNodesPerCell
  /// @brief return number of nodes per cell
  /// @return integer
  Plato::OrdinalType 
  numNodesPerCell() 
  const;

  /// @fn numStateDofsPerNode
  /// @brief return number of state degree of freedom per node
  /// @return integer
  Plato::OrdinalType 
  numStateDofsPerNode() 
  const;

  /// @fn numControlDofsPerNode
  /// @brief return number of control degree of freedom per node
  /// @return integer
  Plato::OrdinalType 
  numControlDofsPerNode() 
  const;

  /// @fn getDofNames
  /// @brief return list with the degree of freedom names
  /// @return standard vector
  std::vector<std::string> 
  getDofNames() 
  const;

  /// @fn getSolutionStateOutputData
  /// @brief get output solutions database
  /// @param [in,out] aSolutions function domain solution database
  /// @return output solutions database
  Plato::Solutions 
  getSolutionStateOutputData(
    const Plato::Solutions & aSolutions
  ) const;

  /// @fn value
  /// @brief return vector function evaluation
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar plato vector
  Plato::ScalarVector
  value(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  );

  /// @fn jacobianState
  /// @brief return jacobian with respect to states
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @param [in] aTranspose applied transpose
  /// @return teuchos reference counter pointer to a plato compressed sparse row (CRS) matrix
  Teuchos::RCP<Plato::CrsMatrixType>
  jacobianState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle,
          bool              aTranspose = true
  );

  /// @fn jacobianNodeState
  /// @brief evaluate jacobian with respect to node states
  /// @param [in] aDatabase  function domain and range database
  /// @param [in] aCycle     scalar, e.g.; time step
  /// @param [in] aTranspose apply transpose
  /// @return teuchos reference counter pointer 
  Teuchos::RCP<Plato::CrsMatrixType>
  jacobianNodeState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle,
          bool              aTranspose
  );

  /// @fn jacobianConfig
  /// @brief return jacobian with respect to configuration
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @param [in] aTranspose applied transpose
  /// @return teuchos reference counter pointer to a plato compressed sparse row (CRS) matrix
  Teuchos::RCP<Plato::CrsMatrixType>
  jacobianConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle,
          bool              aTranspose = true
  );

  /// @fn jacobianControl
  /// @brief return jacobian with respect to controls
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @param [in] aTranspose applied transpose
  /// @return teuchos reference counter pointer to a plato compressed sparse row (CRS) matrix
  Teuchos::RCP<Plato::CrsMatrixType>
  jacobianControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle,
          bool              aTranspose = true
  );
};
    
} // namespace Elliptic


} // namespace Plato
