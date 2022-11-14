#pragma once

#include "Hex8.hpp"
#include "Quad9.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Hex27 Element
*/
/******************************************************************************/
class Hex27
{
  public:

    using Face = Plato::Quad9;
    using C1 = Plato::Hex8;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 27;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 9;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 27;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        constexpr Plato::Scalar wc = 0.17146776406035665885063679070299; // 125/729
        constexpr Plato::Scalar we = 0.27434842249657065416101886512479; // 200/729
        constexpr Plato::Scalar wf = 0.43895747599451301335093944544496; // 320/729
        constexpr Plato::Scalar wb = 0.70233196159122079915704262020881; // 512/729

        return Plato::Array<mNumGaussPoints>({
            wc, we, wc, we, wf, we, wc, we, wc,
            we, wf, we, wf, wb, wf, we, wf, we,
            wc, we, wc, we, wf, we, wc, we, wc
        });
    }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        constexpr Plato::Scalar p = 0.77459666924148340427791481488384; // sqrt(3.0/5.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>({
            -p, -p, -p,    0, -p, -p,    p, -p, -p,
            -p,  0, -p,    0,  0, -p,    p,  0, -p,
            -p,  p, -p,    0,  p, -p,    p,  p, -p,
            -p, -p,  0,    0, -p,  0,    p, -p,  0,
            -p,  0,  0,    0,  0,  0,    p,  0,  0,
            -p,  p,  0,    0,  p,  0,    p,  p,  0,
            -p, -p,  p,    0, -p,  p,    p, -p,  p,
            -p,  0,  p,    0,  0,  p,    p,  0,  p,
            -p,  p,  p,    0,  p,  p,    p,  p,  p
        });
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);

        auto xyz=x*y*z;
        auto yz=y*z;
        auto xy=x*y;
        auto xz=x*z;

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = -(1-x)*(1-y)*(1-z)*xyz/8.0;
        tN(1) =  (1+x)*(1-y)*(1-z)*xyz/8.0;
        tN(2) = -(1+x)*(1+y)*(1-z)*xyz/8.0;
        tN(3) =  (1-x)*(1+y)*(1-z)*xyz/8.0;
        tN(4) =  (1-x)*(1-y)*(1+z)*xyz/8.0;
        tN(5) = -(1+x)*(1-y)*(1+z)*xyz/8.0;
        tN(6) =  (1+x)*(1+y)*(1+z)*xyz/8.0;
        tN(7) = -(1-x)*(1+y)*(1+z)*xyz/8.0;

        tN(8) =  (1-x)*(1+x)*(1-y)*(1-z)*yz/4.0;
        tN(9) = -(1+x)*(1-y)*(1+y)*(1-z)*xz/4.0;
        tN(10)= -(1-x)*(1+x)*(1+y)*(1-z)*yz/4.0;
        tN(11)=  (1-x)*(1-y)*(1+y)*(1-z)*xz/4.0;

        tN(12)=  (1-x)*(1-y)*(1-z)*(1+z)*xy/4.0;
        tN(13)= -(1+x)*(1-y)*(1-z)*(1+z)*xy/4.0;
        tN(14)=  (1+x)*(1+y)*(1-z)*(1+z)*xy/4.0;
        tN(15)= -(1-x)*(1+y)*(1-z)*(1+z)*xy/4.0;

        tN(16)= -(1-x)*(1+x)*(1-y)*(1+z)*yz/4.0;
        tN(17)=  (1+x)*(1-y)*(1+y)*(1+z)*xz/4.0;
        tN(18)=  (1-x)*(1+x)*(1+y)*(1+z)*yz/4.0;
        tN(19)= -(1-x)*(1-y)*(1+y)*(1+z)*xz/4.0;

        tN(20)=  (1-x)*(1+x)*(1-y)*(1+y)*(1-z)*(1+z);

        tN(21)= -(1-x)*(1+x)*(1-y)*(1+y)*(1-z)*z/2.0;
        tN(22)=  (1-x)*(1+x)*(1-y)*(1+y)*(1+z)*z/2.0;

        tN(23)= -(1-x)*x*(1-y)*(1+y)*(1-z)*(1+z)/2.0;
        tN(24)=  (1+x)*x*(1-y)*(1+y)*(1-z)*(1+z)/2.0;

        tN(25)= -(1-x)*(1+x)*(1-y)*y*(1-z)*(1+z)/2.0;
        tN(26)=  (1-x)*(1+x)*(1+y)*y*(1-z)*(1+z)/2.0;

        return tN;
    }

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);
        auto xyz=x*y*z;
        auto yz=y*z;
        auto xy=x*y;
        auto xz=x*z;

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) = -(1-2*x)*(1-y)*(1-z)*yz/8.0;          tG(0,1) = -(1-x)*(1-2*y)*(1-z)*xz/8.0;          tG(0,2) = -(1-x)*(1-y)*(1-2*z)*xy/8.0;
        tG(1,0) =  (1+2*x)*(1-y)*(1-z)*yz/8.0;          tG(1,1) =  (1+x)*(1-2*y)*(1-z)*xz/8.0;          tG(1,2) =  (1+x)*(1-y)*(1-2*z)*xy/8.0;
        tG(2,0) = -(1+2*x)*(1+y)*(1-z)*yz/8.0;          tG(2,1) = -(1+x)*(1+2*y)*(1-z)*xz/8.0;          tG(2,2) = -(1+x)*(1+y)*(1-2*z)*xy/8.0;
        tG(3,0) =  (1-2*x)*(1+y)*(1-z)*yz/8.0;          tG(3,1) =  (1-x)*(1+2*y)*(1-z)*xz/8.0;          tG(3,2) =  (1-x)*(1+y)*(1-2*z)*xy/8.0;
        tG(4,0) =  (1-2*x)*(1-y)*(1+z)*yz/8.0;          tG(4,1) =  (1-x)*(1-2*y)*(1+z)*xz/8.0;          tG(4,2) =  (1-x)*(1-y)*(1+2*z)*xy/8.0;
        tG(5,0) = -(1+2*x)*(1-y)*(1+z)*yz/8.0;          tG(5,1) = -(1+x)*(1-2*y)*(1+z)*xz/8.0;          tG(5,2) = -(1+x)*(1-y)*(1+2*z)*xy/8.0;
        tG(6,0) =  (1+2*x)*(1+y)*(1+z)*yz/8.0;          tG(6,1) =  (1+x)*(1+2*y)*(1+z)*xz/8.0;          tG(6,2) =  (1+x)*(1+y)*(1+2*z)*xy/8.0;
        tG(7,0) = -(1-2*x)*(1+y)*(1+z)*yz/8.0;          tG(7,1) = -(1-x)*(1+2*y)*(1+z)*xz/8.0;          tG(7,2) = -(1-x)*(1+y)*(1+2*z)*xy/8.0;
        tG(8,0) = -(1-y)*(1-z)*xyz/2.0;                 tG(8,1) =  (1-x)*(1+x)*(1-2*y)*(1-z)*z/4.0;     tG(8,2) =  (1-x)*(1+x)*(1-y)*(1-2*z)*y/4.0;
        tG(9,0) = -(1+2*x)*(1-y)*(1+y)*(1-z)*z/4.0;     tG(9,1) =  (1+x)*(1-z)*xyz/2.0;                 tG(9,2) = -(1+x)*(1-y)*(1+y)*(1-2*z)*x/4.0;
        tG(10,0)=  (1+y)*(1-z)*xyz/2.0;                 tG(10,1)= -(1-x)*(1+x)*(1+2*y)*(1-z)*z/4.0;     tG(10,2)= -(1-x)*(1+x)*(1+y)*(1-2*z)*y/4.0;
        tG(11,0)=  (1-2*x)*(1-y)*(1+y)*(1-z)*z/4.0;     tG(11,1)= -(1-x)*(1-z)*xyz/2.0;                 tG(11,2)=  (1-x)*(1-y)*(1+y)*(1-2*z)*x/4.0;
        tG(12,0)=  (1-2*x)*(1-y)*(1-z)*(1+z)*y/4.0;     tG(12,1)=  (1-x)*(1-2*y)*(1-z)*(1+z)*x/4.0;     tG(12,2)= -(1-x)*(1-y)*xyz/2.0;
        tG(13,0)= -(1+2*x)*(1-y)*(1-z)*(1+z)*y/4.0;     tG(13,1)= -(1+x)*(1-2*y)*(1-z)*(1+z)*x/4.0;     tG(13,2)=  (1+x)*(1-y)*xyz/2.0;
        tG(14,0)=  (1+2*x)*(1+y)*(1-z)*(1+z)*y/4.0;     tG(14,1)=  (1+x)*(1+2*y)*(1-z)*(1+z)*x/4.0;     tG(14,2)= -(1+x)*(1+y)*xyz/2.0;
        tG(15,0)= -(1-2*x)*(1+y)*(1-z)*(1+z)*y/4.0;     tG(15,1)= -(1-x)*(1+2*y)*(1-z)*(1+z)*x/4.0;     tG(15,2)=  (1-x)*(1+y)*xyz/2.0;
        tG(16,0)=  (1-y)*(1+z)*xyz/2.0;                 tG(16,1)= -(1-x)*(1+x)*(1-2*y)*(1+z)*z/4.0;     tG(16,2)= -(1-x)*(1+x)*(1-y)*(1+2*z)*y/4.0;
        tG(17,0)=  (1+2*x)*(1-y)*(1+y)*(1+z)*z/4.0;     tG(17,1)= -(1+x)*(1+z)*xyz/2.0;                 tG(17,2)=  (1+x)*(1-y)*(1+y)*(1+2*z)*x/4.0;
        tG(18,0)= -(1+y)*(1+z)*xyz/2.0;                 tG(18,1)=  (1-x)*(1+x)*(1+2*y)*(1+z)*z/4.0;     tG(18,2)=  (1-x)*(1+x)*(1+y)*(1+2*z)*y/4.0;
        tG(19,0)= -(1-2*x)*(1-y)*(1+y)*(1+z)*z/4.0;     tG(19,1)=  (1-x)*(1+z)*xyz/2.0;                 tG(19,2)= -(1-x)*(1-y)*(1+y)*(1+2*z)*x/4.0;
        tG(20,0)= -2*x*(1-y)*(1+y)*(1-z)*(1+z);         tG(20,1)= -2*y*(1-x)*(1+x)*(1-z)*(1+z);         tG(20,2)= -2*z*(1-x)*(1+x)*(1-y)*(1+y);
        tG(21,0)=  (1-y)*(1+y)*(1-z)*xz;                tG(21,1)=  (1-x)*(1+x)*(1-z)*yz;                tG(21,2)= -(1-x)*(1+x)*(1-y)*(1+y)*(1-2*z)/2.0;
        tG(22,0)= -(1-y)*(1+y)*(1+z)*xz;                tG(22,1)= -(1-x)*(1+x)*(1+z)*yz;                tG(22,2)=  (1-x)*(1+x)*(1-y)*(1+y)*(1+2*z)/2.0;
        tG(23,0)= -(1-2*x)*(1-y)*(1+y)*(1-z)*(1+z)/2.0; tG(23,1)=  (1-x)*(1-z)*(1+z)*xy;                tG(23,2)=  (1-x)*(1-y)*(1+y)*xz;
        tG(24,0)=  (1+2*x)*(1-y)*(1+y)*(1-z)*(1+z)/2.0; tG(24,1)= -(1+x)*(1-z)*(1+z)*xy;                tG(24,2)= -(1+x)*(1-y)*(1+y)*xz;
        tG(25,0)=  (1-y)*(1-z)*(1+z)*xy;                tG(25,1)= -(1-x)*(1+x)*(1-2*y)*(1-z)*(1+z)/2.0; tG(25,2)=  (1-x)*(1+x)*(1-y)*yz;
        tG(26,0)= -(1+y)*(1-z)*(1+z)*xy;                tG(26,1)=  (1-x)*(1+x)*(1+2*y)*(1-z)*(1+z)/2.0; tG(26,2)= -(1-x)*(1+x)*(1+y)*yz;

        return tG;
    }
};

} // end namespace Plato
