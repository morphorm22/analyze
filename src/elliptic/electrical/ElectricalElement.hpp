/*
 * ElectricalElement.hpp
 *
 *  Created on: May 23, 2023
 */

#pragma once

#include "ElementBase.hpp"

namespace Plato
{

/// @brief base class for an electrical element
/// @tparam TopoElementTypeT topological element type
/// @tparam NumControls number of control fields
template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class ElectricalElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumNodesPerFace;
    using TopoElementTypeT::mNumSpatialDims;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumDofsPerNode = 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
};
// class ElectricalElement 

}