#pragma once

#include "Bar2.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tri3 Element
*/
/******************************************************************************/
class Tri3
{
  public:
    using Face = Plato::Bar2;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 3;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 2;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 1;

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
