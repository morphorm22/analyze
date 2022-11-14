#pragma once

#include "ElementBase.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
/*! Base class for projection element
*/
/******************************************************************************/
template<
  typename TopoElementTypeT,
  Plato::OrdinalType TotalDofs = TopoElementTypeT::mNumSpatialDims,
  Plato::OrdinalType ProjectionDofOffset = 0,
  Plato::OrdinalType NumProjectionDof = 1,
  Plato::OrdinalType NumControls = 1>
class ProjectionElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
  public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumNodesPerFace;
    using TopoElementTypeT::mNumSpatialDims;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mTotalDofs         = TotalDofs;
    static constexpr Plato::OrdinalType mNumControl        = NumControls;
    static constexpr Plato::OrdinalType mProjectionDof     = ProjectionDofOffset;
    static constexpr Plato::OrdinalType mNumDofsPerNode    = mNumSpatialDims;
    static constexpr Plato::OrdinalType mPressureDofOffset = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerCell    = mNumDofsPerNode * mNumNodesPerCell;

    // this physics can be used with Stabilized functionality in PA.  The
    // following defines the nodal state attributes required by Stabilized
    //
    static constexpr Plato::OrdinalType mNumNodeStatePerNode = NumProjectionDof;
    static constexpr Plato::OrdinalType mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};

} // namespace Stabilized
} // namespace Plato
