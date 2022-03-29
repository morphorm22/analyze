#pragma once

#include "ElementBase.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for mechanics element
*/
/******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumControls = 1>
class MechanicsElement : public ElementType, public ElementBase<ElementType>
{
  public:
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

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
