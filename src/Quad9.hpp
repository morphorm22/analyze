#pragma once

#include "Quad4.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Quad9 Element
*/
/******************************************************************************/
class Quad9
{
  public:
    using C1 = Plato::Quad4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 9;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 3;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 9;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        constexpr Plato::Scalar wc = 0.30864197530864195817557060763647; // 25/81
        constexpr Plato::Scalar we = 0.49382716049382713308091297221836; // 40/81
        constexpr Plato::Scalar wf = 0.79012345679012341292946075554937; // 64/81

        return Plato::Array<mNumGaussPoints>({
            wc, we, wc, we, wf, we, wc, we, wc
        });
    }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        constexpr Plato::Scalar p = 0.77459666924148340427791481488384; // sqrt(3.0/5.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>({
            -p, -p,   0, -p,   p, -p,
            -p,  0,   0,  0,   p,  0,
            -p,  p,   0,  p,   p,  p
        });
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto xy = x*y;

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = (1-x)*(1-y)*xy/4.0;
        tN(1) =-(1+x)*(1-y)*xy/4.0;
        tN(2) = (1+x)*(1+y)*xy/4.0;
        tN(3) =-(1-x)*(1+y)*xy/4.0;
        tN(4) =-(1-x)*(1+x)*(1-y)*y/2.0;
        tN(5) = (1+x)*(1-y)*(1+y)*x/2.0;
        tN(6) = (1-x)*(1+x)*(1+y)*y/2.0;
        tN(7) =-(1-x)*(1-y)*(1+y)*x/2.0;
        tN(8) = (1-x)*(1+x)*(1-y)*(1+y);

        return tN;
    }

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto xy = x*y;

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) = (1-2*x)*(1-y)*y/4.0; tG(0,1) = (1-x)*(1-2*y)*x/4.0;
        tG(1,0) =-(1+2*x)*(1-y)*y/4.0; tG(1,1) =-(1+x)*(1-2*y)*x/4.0;
        tG(2,0) = (1+2*x)*(1+y)*y/4.0; tG(2,1) = (1+x)*(1+2*y)*x/4.0;
        tG(3,0) =-(1-2*x)*(1+y)*y/4.0; tG(3,1) =-(1-x)*(1+2*y)*x/4.0;

        tG(4,0) = (1-y)*xy;                tG(4,1) =-(1-x)*(1+x)*(1-2*y)/2.0;
        tG(5,0) = (1+2*x)*(1-y)*(1+y)/2.0; tG(5,1) =-(1+x)*xy;
        tG(6,0) =-(1+y)*xy;                tG(6,1) = (1-x)*(1+x)*(1+2*y)/2.0;
        tG(7,0) =-(1-2*x)*(1-y)*(1+y)/2.0; tG(7,1) = (1-x)*xy;
        tG(8,0) =-2*(1-y)*(1+y)*x;         tG(8,1) =-2*(1-x)*(1+x)*y;

        return tG;
    }

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION static 
    ScalarType differentialMeasure(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        ScalarType ax = aJacobian(0,1)*aJacobian(1,2)-aJacobian(0,2)*aJacobian(1,1);
        ScalarType ay = aJacobian(0,2)*aJacobian(1,0)-aJacobian(0,0)*aJacobian(1,2);
        ScalarType az = aJacobian(0,0)*aJacobian(1,1)-aJacobian(0,1)*aJacobian(1,0);

        return sqrt(ax*ax+ay*ay+az*az);
    }

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION static 
    Plato::Array<mNumSpatialDims+1, ScalarType>
    differentialVector(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        Plato::Array<mNumSpatialDims+1, ScalarType> tReturnVec;
        tReturnVec(0) = aJacobian(0,1)*aJacobian(1,2)-aJacobian(0,2)*aJacobian(1,1);
        tReturnVec(1) = aJacobian(0,2)*aJacobian(1,0)-aJacobian(0,0)*aJacobian(1,2);
        tReturnVec(2) = aJacobian(0,0)*aJacobian(1,1)-aJacobian(0,1)*aJacobian(1,0);

        return tReturnVec;
    }
};

} // end namespace Plato
