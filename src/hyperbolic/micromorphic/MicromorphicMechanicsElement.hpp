#pragma once

#include "ElementBase.hpp"

namespace Plato
{

namespace Hyperbolic
{

template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class MicromorphicMechanicsElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
  public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumNodesPerFace;
    using TopoElementTypeT::mNumSpatialDims;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumFullTerms    = (mNumSpatialDims == 3) ? 9 :
                                             ((mNumSpatialDims == 2) ? 4 :
                                            (((mNumSpatialDims == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType mNumVoigtTerms   = (mNumSpatialDims == 3) ? 6 :
                                             ((mNumSpatialDims == 2) ? 3 :
                                            (((mNumSpatialDims == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType mNumSkwTerms     = (mNumSpatialDims == 3) ? 3 :
                                             ((mNumSpatialDims == 2) ? 1 :
                                            (((mNumSpatialDims == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mNumDofsPerNode  = mNumSpatialDims + mNumFullTerms;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;

};

} 

} 

