/*
 *  NeumannBoundaryConditionBase.hpp
 *
 *  Created on: July 5, 2023
 */


#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato
{

template<Plato::OrdinalType NumForceDof>
class NeumannBoundaryConditionBase
{
protected:
  /// @brief side set name
  std::string mSideSetName;
  /// @brief force magnitudes
  Plato::Array<NumForceDof> mFlux;

public:
  virtual 
  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  ) const = 0;

  virtual
  void
  flux(
    const Plato::Array<NumForceDof> & aFlux
  ) = 0;

  std::string
  sideset() 
  const
  { return mSideSetName; }
};

} // namespace Plato
