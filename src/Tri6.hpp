#pragma once

#include "Bar3.hpp"

namespace Plato
{

/******************************************************************************/
/*! Tri6 Element
 *
 * See Klaus-Jurgen Bathe, "Finite Element Proceedures", 2006, pg 374.
 *
 * \brief Gauss point coordinates and weights are derived on integration
 *     domain 0<=t<=1.
 *
*/
/******************************************************************************/
class Tri6
{
  public:
    using Face = Plato::Bar3;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 6;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 3;

    static constexpr Plato::OrdinalType mNumFacesPerCell = 3;
    static constexpr Plato::OrdinalType mNumNodesPerFace       = Face::mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumGaussPointsPerFace = Face::mNumGaussPoints;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        return Plato::Array<mNumGaussPoints>({Plato::Scalar(1.0)/6.0, Plato::Scalar(1.0)/6.0, Plato::Scalar(1.0)/6.0});
    }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>({
            Plato::Scalar(2.0)/3, Plato::Scalar(1.0)/6, 
            Plato::Scalar(1.0)/6, Plato::Scalar(2.0)/3, 
            Plato::Scalar(1.0)/6, Plato::Scalar(1.0)/6 
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
        constexpr Plato::Scalar tPt1 = 0.11270166537925829786104259255808; // -0.5*sqrt(1.0/3.0) + 0.5
        constexpr Plato::Scalar tPt2 = 0.5; // -0.5*0.0 + 0.5
        constexpr Plato::Scalar tPt3 = 0.88729833462074170213895740744192; // 0.5*sqrt(1.0/3.0) + 0.5
        return Plato::Matrix<mNumFacesPerCell,mNumSpatialDims*mNumGaussPointsPerFace>({
            /*GP1=*/tPt1  ,tZero, /*GP2=*/tPt2  ,tZero, /*GP3=*/tPt3  ,tZero,
            /*GP1=*/tPt1  ,tPt3  , /*GP2=*/tPt2  ,tPt2  , /*GP3=*/tPt3  ,tPt1  ,
            /*GP1=*/tZero,tPt1  , /*GP2=*/tZero,tPt2  , /*GP3=*/tZero,tPt3
        });
    }

    /******************************************************************************/
    /*! \fn getFaceCubPoints
     *
     * \brief Returns the Gauss point weights associated with Gauss points along
     *     the element edge defined on integration domain 0<=t<=1. The length
     *     of the edge equals one. Therefore, the Gauss weights are derived by
     *     applying a factor of 0.5 to the Gauss weights for a Bar2 element defined
     *     on integration domain -1<=t<=1.
    */
    /******************************************************************************/
    static inline Plato::Array<mNumGaussPointsPerFace>
    getFaceCubWeights()
    {
        constexpr Plato::Scalar tW1 = Plato::Scalar(5.0)/18;
        constexpr Plato::Scalar tW2 = Plato::Scalar(8.0)/18;
        return Plato::Array<mNumGaussPointsPerFace>( {tW1, tW1, tW2} );
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto x2=x*x;
        auto y2=y*y;
        auto xy=x*y;

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = 1-3*(x+y)+2*(x2+y2)+4*xy;
        tN(1) = 2*x2-x;
        tN(2) = 2*y2-y;
        tN(3) = 4*(x-xy-x2);
        tN(4) = 4*xy;
        tN(5) = 4*(y-xy-y2);

        return tN;
    }

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) =  4*(x+y)-3    ; tG(0,1) =  4*(x+y)-3;
        tG(1,0) =  4*x-1        ; tG(1,1) =  0;
        tG(2,0) =  0            ; tG(2,1) =  4*y-1;
        tG(3,0) =  4-8*x-4*y    ; tG(3,1) = -4*x;
        tG(4,0) =  4*y          ; tG(4,1) =  4*x;
        tG(5,0) = -4*y          ; tG(5,1) =  4-4*x-8*y;

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
