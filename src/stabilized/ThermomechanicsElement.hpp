#pragma once

#include "ElementBase.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
/*! Base class for two-field thermomechanics
*/
/******************************************************************************/
template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class ThermomechanicsElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
  public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumSpatialDims;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumVoigtTerms = (mNumSpatialDims == 3) ? 6 :
                                                        ((mNumSpatialDims == 2) ? 3 :
                                                       (((mNumSpatialDims == 1) ? 1 : 0)));

    // degree-of-freedom attributes
    static constexpr auto mNumControl        = NumControls;

    static constexpr auto mTDofOffset        = mNumSpatialDims + 1;
    static constexpr auto mNumDofsPerNode    = mNumSpatialDims + 2;
    static constexpr auto mPressureDofOffset = mNumSpatialDims;
    static constexpr auto mNumDofsPerCell    = mNumDofsPerNode * mNumNodesPerCell;

    // this physics can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    //
    static constexpr auto mNumNodeStatePerNode = mNumSpatialDims;
    static constexpr auto mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell;
    static constexpr auto mNumLocalDofsPerCell = 0;
};


} // namespace Stabilized
} // namespace Plato
