#pragma once

#include "Tri3.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tet4 Element
*/
/******************************************************************************/
class Tet4
{
  public:

    using Face = Plato::Tri3;
    using C1 = Plato::Tet4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 4;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 3;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<1>
    getCubWeights() { return Plato::Array<1>({Plato::Scalar(1.0)/6}); }

    static inline Plato::Matrix<1,mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<1,mNumSpatialDims>({
            Plato::Scalar(1.0)/3, Plato::Scalar(1.0)/3, Plato::Scalar(1.0)/3
        });
    }

    DEVICE_TYPE static inline Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = Plato::Scalar(1) - x - y - z;
        tN(1) = x;
        tN(2) = y;
        tN(3) = z;

        return tN;
    }

    DEVICE_TYPE static inline Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) = -1; tG(0,1) = -1; tG(0,2) = -1;
        tG(1,0) =  1; tG(1,1) =  0; tG(1,2) =  0;
        tG(2,0) =  0; tG(2,1) =  1; tG(2,2) =  0;
        tG(3,0) =  0; tG(3,1) =  0; tG(3,2) =  1;

        return tG;
    }
};

} // end namespace Plato
