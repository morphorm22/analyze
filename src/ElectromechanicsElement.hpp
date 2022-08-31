#pragma once

#include "ElementBase.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based electromechanics
*/
/******************************************************************************/
template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class ElectromechanicsElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
  public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumSpatialDims;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumVoigtTerms   = (mNumSpatialDims == 3) ? 6 :
                                                          ((mNumSpatialDims == 2) ? 3 :
                                                         (((mNumSpatialDims == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mEDofOffset      = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerNode  = mNumSpatialDims + 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;

};

} // namespace Plato
