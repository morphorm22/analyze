#pragma once

#include "Bar2.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Quad4 Element
*/
/******************************************************************************/
class Quad4
{
  public:
    using Face = Plato::Bar2;
    using C1 = Plato::Quad4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 4;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 2;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 4;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        return Plato::Array<mNumGaussPoints>({
            Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0)
        });
    }

    static inline Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        const Plato::Scalar sqt = 0.57735026918962584208117050366127; // sqrt(1.0/3.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>({
            -sqt, -sqt,
             sqt, -sqt,
             sqt,  sqt,
            -sqt,  sqt
        });
    }

    KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = (1-x)*(1-y)/4.0;
        tN(1) = (1+x)*(1-y)/4.0;
        tN(2) = (1+x)*(1+y)/4.0;
        tN(3) = (1-x)*(1+y)/4.0;

        return tN;
    }

    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) = -(1-y)/4.0; tG(0,1) = -(1-x)/4.0;
        tG(1,0) =  (1-y)/4.0; tG(1,1) = -(1+x)/4.0;
        tG(2,0) =  (1+y)/4.0; tG(2,1) =  (1+x)/4.0;
        tG(3,0) = -(1+y)/4.0; tG(3,1) =  (1-x)/4.0;

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
