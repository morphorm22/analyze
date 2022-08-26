#pragma once

#include "Tri6.hpp"
#include "Tet4.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tet10 Element
*/
/******************************************************************************/
class Tet10
{
  public:

    using Face = Plato::Tri6;
    using C1 = Plato::Tet4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 10;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 6;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 4;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        return Plato::Array<mNumGaussPoints>
                 ({Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0});
    }

    static inline Plato::Matrix<mNumGaussPoints, mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<mNumGaussPoints, mNumSpatialDims>({
            0.585410196624969, 0.138196601125011, 0.138196601125011,
            0.138196601125011, 0.585410196624969, 0.138196601125011,
            0.138196601125011, 0.138196601125011, 0.585410196624969,
            0.138196601125011, 0.138196601125011, 0.138196601125011
        });
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);

        Plato::Array<mNumNodesPerCell> tN;

        auto tCon = (x+y+z-1.0);
        tN(0) =  tCon*(2.0*x+2.0*y+2.0*z-1);
        tN(1) =  x*(2.0*x-1.0);
        tN(2) =  y*(2.0*y-1.0);
        tN(3) =  z*(2.0*z-1.0);
        tN(4) = -4.0*x*tCon;
        tN(5) =  4.0*x*y;
        tN(6) = -4.0*y*tCon;
        tN(7) = -4.0*z*tCon;
        tN(8) =  4.0*x*z;
        tN(9) =  4.0*y*z;

        return tN;
    }

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) = (x+y+z)*4.0-3.0    ; tG(0,1) = (x+y+z)*4.0-3.0    ; tG(0,2) = (x+y+z)*4.0-3.0;
        tG(1,0) = x*4.0-1.0          ; tG(1,1) = 0.0                ; tG(1,2) = 0.0;
        tG(2,0) = 0.0                ; tG(2,1) = y*4.0-1.0          ; tG(2,2) = 0.0;
        tG(3,0) = 0.0                ; tG(3,1) = 0.0                ; tG(3,2) = z*4.0-1.0;
        tG(4,0) = -(x*2.0+y+z-1)*4.0 ; tG(4,1) = -x*4.0             ; tG(4,2) = -x*4.0;
        tG(5,0) = y*4.0              ; tG(5,1) = x*4.0              ; tG(5,2) = 0.0;
        tG(6,0) = -y*4.0             ; tG(6,1) = -(x+2.0*y+z-1)*4.0 ; tG(6,2) = -y*4.0;
        tG(7,0) = -z*4.0             ; tG(7,1) = -z*4.0             ; tG(7,2) = -(x+y+z*2.0-1)*4.;
        tG(8,0) = z*4.0              ; tG(8,1) = 0.0                ; tG(8,2) = x*4.;
        tG(9,0) = 0.0                ; tG(9,1) = z*4.0              ; tG(9,2) = y*4.;

        return tG;
    }
};

} // end namespace Plato
