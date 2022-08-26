#pragma once

#include "Quad4.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Hex8 Element
*/
/******************************************************************************/
class Hex8
{
  public:

    using Face = Plato::Quad4;
    using C1 = Plato::Hex8;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 8;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 4;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 8;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        return Plato::Array<mNumGaussPoints>({
            Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0),
            Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0)
        });
    }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        const Plato::Scalar sqt = 0.57735026918962584208117050366127; // sqrt(1.0/3.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>({
            -sqt, -sqt, -sqt,
             sqt, -sqt, -sqt,
             sqt,  sqt, -sqt,
            -sqt,  sqt, -sqt,
            -sqt, -sqt,  sqt,
             sqt, -sqt,  sqt,
             sqt,  sqt,  sqt,
            -sqt,  sqt,  sqt,
        });
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = (1-x)*(1-y)*(1-z)/8.0;
        tN(1) = (1+x)*(1-y)*(1-z)/8.0;
        tN(2) = (1+x)*(1+y)*(1-z)/8.0;
        tN(3) = (1-x)*(1+y)*(1-z)/8.0;
        tN(4) = (1-x)*(1-y)*(1+z)/8.0;
        tN(5) = (1+x)*(1-y)*(1+z)/8.0;
        tN(6) = (1+x)*(1+y)*(1+z)/8.0;
        tN(7) = (1-x)*(1+y)*(1+z)/8.0;

        return tN;
    }

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) = -(1-y)*(1-z)/8.0; tG(0,1) = -(1-x)*(1-z)/8.0; tG(0,2) = -(1-x)*(1-y)/8.0;
        tG(1,0) =  (1-y)*(1-z)/8.0; tG(1,1) = -(1+x)*(1-z)/8.0; tG(1,2) = -(1+x)*(1-y)/8.0;
        tG(2,0) =  (1+y)*(1-z)/8.0; tG(2,1) =  (1+x)*(1-z)/8.0; tG(2,2) = -(1+x)*(1+y)/8.0;
        tG(3,0) = -(1+y)*(1-z)/8.0; tG(3,1) =  (1-x)*(1-z)/8.0; tG(3,2) = -(1-x)*(1+y)/8.0;
        tG(4,0) = -(1-y)*(1+z)/8.0; tG(4,1) = -(1-x)*(1+z)/8.0; tG(4,2) =  (1-x)*(1-y)/8.0;
        tG(5,0) =  (1-y)*(1+z)/8.0; tG(5,1) = -(1+x)*(1+z)/8.0; tG(5,2) =  (1+x)*(1-y)/8.0;
        tG(6,0) =  (1+y)*(1+z)/8.0; tG(6,1) =  (1+x)*(1+z)/8.0; tG(6,2) =  (1+x)*(1+y)/8.0;
        tG(7,0) = -(1+y)*(1+z)/8.0; tG(7,1) =  (1-x)*(1+z)/8.0; tG(7,2) =  (1-x)*(1+y)/8.0;

        return tG;
    }
};

} // end namespace Plato
