/*
 *  NitscheBase.hpp
 *
 *  Created on: July 6, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/// @class NitscheBase
/// @brief parent class for nitsche's method
class NitscheBase
{
public: 
  /// @fn evaluate
  /// @brief evaluate nitsche's boundary condition
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     range and domain database
  /// @param [in]     aCycle        scalar input
  /// @param [in]     aScale        scalar multiplier
  virtual 
  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  ) = 0;
};

} // namespace Plato
