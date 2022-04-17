#pragma once

#include "ElementBase.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based helmholtz
*/
/******************************************************************************/
template<typename TopoElementTypeT>
class HelmholtzElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{ 
  public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumNodesPerFace;
    using TopoElementTypeT::mNumSpatialDims;
    using TopoElementTypeT::mNumSpatialDimsOnFace;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumDofsPerNode  = 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = 1;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;

};

}
