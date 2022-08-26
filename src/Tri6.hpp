#pragma once

namespace Plato {

/******************************************************************************/
/*! Tri6 Element

    See Klaus-Jurgen Bathe, "Finite Element Proceedures", 2006, pg 374.
*/
/******************************************************************************/
class Tri6
{
  public:
    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 6;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 3;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 3;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights() { return Plato::Array<mNumGaussPoints>({Plato::Scalar(1.0)/6.0, Plato::Scalar(1.0)/6.0, Plato::Scalar(1.0)/6.0}); }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>({
            Plato::Scalar(2.0)/3, Plato::Scalar(1.0)/6, 
            Plato::Scalar(1.0)/6, Plato::Scalar(2.0)/3, 
            Plato::Scalar(1.0)/6, Plato::Scalar(1.0)/6 
        });
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
