#pragma once

#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Quad4 Element
*/
/******************************************************************************/
class Quad4
{
  public:

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 4;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 2;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<4>
    getCubWeights()
    {
        return Plato::Array<4>({
            Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0)
        });
    }

    static inline Plato::Matrix<4,mNumSpatialDims>
    getCubPoints()
    {
        const Plato::Scalar sqt = 0.57735026918962584208117050366127; // sqrt(1.0/3.0)
        return Plato::Matrix<4,mNumSpatialDims>({
            -sqt, -sqt,
             sqt, -sqt,
             sqt,  sqt,
            -sqt,  sqt
        });
    }

    DEVICE_TYPE static inline Plato::Array<mNumNodesPerCell>
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

    DEVICE_TYPE static inline Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
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
    DEVICE_TYPE static inline
    ScalarType differentialMeasure(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        auto ax = aJacobian(0,1)*aJacobian(1,2)-aJacobian(0,2)*aJacobian(1,1);
        auto ay = aJacobian(0,2)*aJacobian(1,0)-aJacobian(0,0)*aJacobian(1,2);
        auto az = aJacobian(0,0)*aJacobian(1,1)-aJacobian(0,1)*aJacobian(1,0);

        return sqrt(ax*ax+ay*ay+az*az);
    }
};

} // end namespace Plato
