#pragma once

#include "ElementBase.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for mechanics element
*/
/******************************************************************************/
template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class MechanicsElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
  public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumNodesPerFace;
    using TopoElementTypeT::mNumSpatialDims;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumVoigtTerms   = (mNumSpatialDims == 3) ? 6 :
                                                          ((mNumSpatialDims == 2) ? 3 :
                                                         (((mNumSpatialDims == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType mNumDofsPerNode  = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;


    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};

}
