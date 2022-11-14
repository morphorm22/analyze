#pragma once

#include "ElementBase.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for geometrical element
*/
/******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumControls = 1>
class GeometricalElement : public ElementType, public ElementBase<ElementType>
{
  public:
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumNodesPerFace;
    using ElementType::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumControl = NumControls;
};

}
