/*
 *  ResidualBase.hpp
 *
 *  Created on: June 6, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "Solutions.hpp"
#include "SpatialModel.hpp"

#include "elliptic/evaluators/problem/SupportedEllipticProblemOptions.hpp"

namespace Plato
{

/// @class ResidualBase
/// @brief residual evaluator base class
class ResidualBase
{
protected:
  /// @brief containes mesh and model information
  const Plato::SpatialDomain & mSpatialDomain; 
  /// @brief output database
  Plato::DataMap & mDataMap;
  /// @brief database for degree of freedom names
  std::vector<std::string>   mDofNames;       

public:
  /// @brief typename for residual base class
  using AbstractType = Plato::ResidualBase;

  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  explicit ResidualBase(
    const Plato::SpatialDomain & aSpatialDomain,
          Plato::DataMap       & aDataMap
  ) :
    mSpatialDomain(aSpatialDomain),
    mDataMap(aDataMap)
  {}

  /// @brief class destructor
  virtual ~ResidualBase(){}

  /// @fn getMesh
  /// @brief get mesh database
  /// @return shared pointer to mesh database
  decltype(mSpatialDomain.Mesh) 
  getMesh() 
  const
  {
    return (mSpatialDomain.Mesh);
  }

  /// @fn getDofNames
  /// @brief return database of degree of freedom names
  /// @return 
  const decltype(mDofNames)& 
  getDofNames() 
  const
  {
    return mDofNames;
  }

  /// @fn type
  /// @brief get residual type
  /// @return residual_t enum
  virtual
  Plato::Elliptic::residual_t
  type() 
  const =0;

  /// @fn getSolutionStateOutputData
  /// @brief get output solutions database
  /// @param [in] aSolutions function domain solution database
  /// @return output solutions database
  virtual 
  Plato::Solutions 
  getSolutionStateOutputData(
    const Plato::Solutions & aSolutions
  ) const = 0;

  /// @fn evaluate
  /// @brief evaluate internal forces, pure virtual function
  /// @param [in,out] aWorkSets domain and range workset database
  /// @param [in]     aCycle    scalar
  virtual 
  void 
  evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) const = 0;

  /// @fn evaluateBoundary
  /// @brief evaluate boundary forces, pure virtual function
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        scalar
  virtual 
  void 
  evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) const = 0;
};
// class abstract residual

} // namespace Plato
