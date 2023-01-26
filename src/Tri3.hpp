#pragma once

#include "Bar2.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Tri3 Element
 *
 * \brief Gauss point coordinates and weights are derived on integration
 *     domain 0<=t<=1.
 *
*/
/******************************************************************************/
class Tri3
{
  public:
    using Face = Plato::Bar2;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 3;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 1;

    static constexpr Plato::OrdinalType mNumFacesPerCell       = 3;
    static constexpr Plato::OrdinalType mNumNodesPerFace       = Face::mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumGaussPointsPerFace = Face::mNumGaussPoints;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights() { return Plato::Array<mNumGaussPoints>({Plato::Scalar(1)/2}); }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>({
            Plato::Scalar(1)/3, Plato::Scalar(1)/3
        });
    }

    /******************************************************************************/
    /*! \fn getFaceCubPoints
     *
     * \brief Gauss point coordinates and weights are derived on integration
     *     domain 0<=t<=1, which requires the following linear mapping
     *
     *     \hat{\xi}=\left(\frac{b-a}{2}\right)\xi + \left(\frac{b+a}{2}\right)
     *
     *     to map the Gauss points for a Bar2 element from domain -1<=t<=1 to
     *     domain 0<=t<=1.
    */
    /******************************************************************************/
    static inline Plato::Matrix<mNumFacesPerCell,mNumSpatialDims*mNumGaussPointsPerFace>
    getFaceCubPoints()
    {
        constexpr Plato::Scalar tZero = 0.0;
        constexpr Plato::Scalar tPt1 = 0.21132486540518707895941474816937; // -0.5*sqrt(1.0/3.0)+0.5
        constexpr Plato::Scalar tPt2 = 0.78867513459481292104058525183063; //  0.5*sqrt(1.0/3.0)+0.5
        return Plato::Matrix<mNumFacesPerCell,mNumSpatialDims*mNumGaussPointsPerFace>({
            /*GP1=*/tPt1  ,tZero, /*GP2=*/tPt2  ,tZero,
            /*GP1=*/tPt1  ,tPt2  , /*GP2=*/tPt2  ,tPt1,
            /*GP1=*/tZero,tPt1  , /*GP2=*/tZero,tPt2
        });
    }

    /******************************************************************************/
    /*! \fn getFaceCubPoints
     *
     * \brief Returns the Gauss point weights associated with Gauss points along
     *     the element edge defined on integration domain 0<=t<=1. The length
     *     of the edge equals one. Therefore, the Gauss weights are derived by
     *     applying a factor of 0.5 to the Gauss weights for a Bar2 element defined
     *     on integration domain -1<=t<=1. The sum of the weights must equal
     *     the length of the bar element, i.e., 1.0.
    */
    /******************************************************************************/
    static inline Plato::Array<mNumGaussPointsPerFace>
    getFaceCubWeights()
    {
        return Plato::Array<mNumGaussPointsPerFace>({
            Plato::Scalar(0.5), Plato::Scalar(0.5)
        });
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = 1-x-y;
        tN(1) = x;
        tN(2) = y;

        return tN;
    }

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) =-1; tG(0,1) =-1;
        tG(1,0) = 1; tG(1,1) = 0;
        tG(2,0) = 0; tG(2,1) = 1;

        return tG;
    }

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION static 
    ScalarType
    differentialMeasure(
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
