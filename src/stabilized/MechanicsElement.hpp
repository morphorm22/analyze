#pragma once

#include "ElementBase.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
/*! Base class for two-field Mechanics element
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

    static constexpr Plato::OrdinalType mNumVoigtTerms = (mNumSpatialDims == 3) ? 6 :
                                                        ((mNumSpatialDims == 2) ? 3 :
                                                       (((mNumSpatialDims == 1) ? 1 : 0)));

    // degree-of-freedom attributes
    static constexpr auto mNumControl        = NumControls;                        /*!< number of controls */
    static constexpr auto mNumDofsPerNode    = mNumSpatialDims + 1;                /*!< number of degrees of freedom per node { disp_x, disp_y, disp_z, pressure} */
    static constexpr auto mPressureDofOffset = mNumSpatialDims;                    /*!< pressure degree of freedom offset */
    static constexpr auto mNumDofsPerCell    = mNumDofsPerNode * mNumNodesPerCell; /*!< number of degrees of freedom per cell */

    // this element can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    static constexpr auto mNumNodeStatePerNode = mNumSpatialDims;                         /*!< number of node states, i.e. pressure gradient, dofs per node */
    static constexpr auto mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell; /*!< number of node states, i.e. pressure gradient, dofs  per cell */
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};

} // namespace Stabilized
} // namespace Plato
