/*
 * Bar3.hpp
 *
 *  Created on: Jan 22, 2023
 */

#include "PlatoMathTypes.hpp"

#pragma once

namespace Plato
{

/******************************************************************************/
/*! Bar3 Element (Quadratic)
 *
 * \brief Gauss point coordinates and weights are derived on integration
 *     domain -1<=t<=1
*/
/******************************************************************************/
class Bar3
{
  public:

    static constexpr Plato::OrdinalType mNumSpatialDims  = 1;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 3;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 1;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        constexpr Plato::Scalar tW1 = Plato::Scalar(5.0)/9;
        constexpr Plato::Scalar tW2 = Plato::Scalar(8.0)/9;
        return Plato::Array<mNumGaussPoints>( {tW1, tW1, tW2} );
    }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        constexpr Plato::Scalar tPt1 = 0.77459666924148340427791481488384; // sqrt(3.0/5.0)
        constexpr Plato::Scalar tPt2 = 0.0;
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>( {tPt1, -tPt1, tPt2} );
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);

        Plato::Array<mNumNodesPerCell> tN;

        constexpr Plato::Scalar tHalf = 0.5;
        constexpr Plato::Scalar tOne  = 1.0;
        tN(0) = tHalf * ( x*x - x );
        tN(1) = tHalf * ( x*x + x );
        tN(2) = ( tOne - x*x );

        return tN;
    }

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        constexpr Plato::Scalar tHalf = 0.5;
        constexpr Plato::Scalar tOne  = 1.0;
        constexpr Plato::Scalar tTwo  = 2.0;
        tG(0,0) = tHalf * ( tTwo*x - tOne );
        tG(1,0) = tHalf * ( tTwo*x + tOne );
        tG(2,0) = -tTwo*x;

        return tG;
    }

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION static ScalarType
    differentialMeasure(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        ScalarType ax = aJacobian(0,0);
        ScalarType ay = aJacobian(0,1);

        return sqrt(ax*ax+ay*ay);
    }

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION static
    Plato::Array<mNumSpatialDims+1, ScalarType>
    differentialVector(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        Plato::Array<mNumSpatialDims+1, ScalarType> tReturnVec;
        tReturnVec(0) = aJacobian(0,0);
        tReturnVec(1) = aJacobian(0,1);

        return tReturnVec;
    }
};
// class Bar3

}
// namespace Plato
