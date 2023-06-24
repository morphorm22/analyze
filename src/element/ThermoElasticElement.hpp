/*
 * ThermoElasticElement.hpp
 *
 *  Created on: June 16, 2023
 */

#pragma once

#include "ElementBase.hpp"

namespace Plato
{

/// @class ThermoElasticElement
/// @brief base class for thermo-elastic element
/// @tparam TopoElementTypeT topological element type
/// @tparam NumControls      number of control degree of freedom per node
template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class ThermoElasticElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
public:
  /// @brief number of nodes per cell
  using TopoElementTypeT::mNumNodesPerCell;
  /// @brief number of spatial dimensions
  using TopoElementTypeT::mNumSpatialDims;
  /// @brief topological element type
  using TopoElementType = TopoElementTypeT;
  /// @brief number of stress-strain components
  static constexpr Plato::OrdinalType mNumVoigtTerms = (mNumSpatialDims == 3) ? 6 :
                                                       ((mNumSpatialDims == 2) ? 3 :
                                                       (((mNumSpatialDims == 1) ? 1 : 0)));
  /// @brief number of displacement degrees of freedom per node
  static constexpr Plato::OrdinalType mNumDofsPerNode = mNumSpatialDims;
  /// @brief number of displacement degrees of freedom per cell
  static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell;
  /// @brief number of control degrees of freedom per node
  static constexpr Plato::OrdinalType mNumControl = NumControls;
  /// @brief number of temperature degrees of freedom per node
  static constexpr Plato::OrdinalType mNumNodeStatePerNode = 1;
  /// @brief number of temperature degrees of freedom per cell
  static constexpr Plato::OrdinalType mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell;
  /// @brief number of local state degrees of freedom per cell
  static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};

}