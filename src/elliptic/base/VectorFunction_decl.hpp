/*
 * VectorFunction_decl.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

#include <string>
#include <memory>
#include <unordered_map>

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
/// @tparam PhysicsType defines physics-based quantity of interests 
template<typename PhysicsType>
class VectorFunction : public VectorFunctionBase
{
private:
  /// @brief topological element type
  using ElementType = typename PhysicsType::ElementType;
  /// @brief number of nodes per element
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of nodes per element face
  static constexpr auto mNumNodesPerFace = ElementType::mNumNodesPerFace;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of control degree of freedoms per node
  static constexpr auto mNumControl      = ElementType::mNumControl;
  /// @brief number of configuration degress of freedom per element
  static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;
  /// @brief scalar types for a given evaluation type
  using ResidualEvalType  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using JacobianUEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
  using JacobianXEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
  using JacobianZEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
  /// @brief domain (element block) to residual map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mResiduals;
  /// @brief domain (element block) to jacobian with respect to states map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mJacobiansU;
  /// @brief domain (element block) to jacobian with respect to configuration map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mJacobiansX;
  /// @brief domain (element block) to jacobian with respect to controls map
  std::unordered_map<std::string,std::shared_ptr<Plato::ResidualBase>> mJacobiansZ;
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

  /// @fn numDofs
  /// @brief return number of degress of freedom per node
  /// @return integer 
  Plato::OrdinalType 
  numDofs() 
  const;

  /// @fn getDofNames
  /// @brief return list of degree of freedom names
  /// @return standard vector
  std::vector<std::string> 
  getDofNames() 
  const;

  /// @fn value
  /// @brief return vector function evaluation
  /// @param [in] aDatabase output database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar plato vector
  Plato::ScalarVector
  value(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  );

  /// @fn jacobianState
  /// @brief return jacobian with respect to states
  /// @param [in] aDatabase  output database
  /// @param [in] aCycle     scalar, e.g.; time step
  /// @param [in] aTranspose applied transpose
  /// @return teuchos reference counter pointer to a plato compressed sparse row (CRS) matrix
  Teuchos::RCP<Plato::CrsMatrixType>
  jacobianState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle,
          bool              aTranspose = true
  );

  /// @fn jacobianConfig
  /// @brief return jacobian with respect to configuration
  /// @param [in] aDatabase  output database
  /// @param [in] aCycle     scalar, e.g.; time step
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
  /// @param [in] aDatabase  output database
  /// @param [in] aCycle     scalar, e.g.; time step
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
