#pragma once

#include "Tri3.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tet4 Element
 *
 * \brief Gauss point coordinates and weights are derived on integration
 *     domain 0<=t<=1.
 *
*/
/******************************************************************************/
class Tet4
{
  public:

    using Face = Plato::Tri3;
    using C1 = Plato::Tet4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 4;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 1;

    static constexpr Plato::OrdinalType mNumFacesPerCell       = 4;
    static constexpr Plato::OrdinalType mNumNodesPerFace       = Face::mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumGaussPointsPerFace = Face::mNumGaussPoints;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights() { return Plato::Array<mNumGaussPoints>({Plato::Scalar(1.0)/6}); }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        constexpr Plato::Scalar tPt = Plato::Scalar(1.0)/4;
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>( {tPt, tPt, tPt} );
    }

    static inline Plato::Matrix<mNumFacesPerCell,mNumSpatialDims*mNumGaussPointsPerFace>
    getFaceCubPoints()
    {
        constexpr Plato::Scalar tZero = 0.0;
        constexpr Plato::Scalar tPt = Plato::Scalar(1.0)/3;
        return Plato::Matrix<mNumFacesPerCell,mNumSpatialDims*mNumGaussPointsPerFace>({
            /*GP1=*/tPt  , tZero, tPt,/*Face 1*/
            /*GP1=*/tPt  , tPt  , tPt,/*Face 2*/
            /*GP1=*/tZero, tPt  , tPt,/*Face 3*/
            /*GP1=*/tPt  , tPt  , tZero/*Face 4*/
        });
    }

    static inline Plato::Array<mNumGaussPointsPerFace>
    getFaceCubWeights()
    {
        return Face::getCubWeights();
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
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

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
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
