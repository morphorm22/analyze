#pragma once

#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Bar2 Element
*/
/******************************************************************************/
class Bar2
{
  public:

    static constexpr Plato::OrdinalType mNumSpatialDims  = 1;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 2;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 1;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static inline Plato::Array<2>
    getCubWeights()
    {
        return Plato::Array<2>({
            Plato::Scalar(1.0), Plato::Scalar(1.0)
        });
    }

    static inline Plato::Matrix<2,mNumSpatialDims>
    getCubPoints()
    {
        const Plato::Scalar sqt = 0.57735026918962584208117050366127; // sqrt(1.0/3.0)
        return Plato::Matrix<2,mNumSpatialDims>({ -sqt,  sqt });
    }

    DEVICE_TYPE static inline Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = (1-x)/2.0;
        tN(1) = (1+x)/2.0;

        return tN;
    }

    DEVICE_TYPE static inline Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) = Plato::Scalar(-1)/2.0;
        tG(1,0) = Plato::Scalar(1)/2.0;

        return tG;
    }

    template<typename ScalarType>
    DEVICE_TYPE static inline
    ScalarType
    differentialMeasure(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        auto ax = aJacobian(0,0)*aJacobian(0,0);
        auto ay = aJacobian(0,1)*aJacobian(0,1);

        return sqrt(ax*ax+ay*ay);
    }

    template<typename ScalarType>
    DEVICE_TYPE static inline
    Plato::Array<mNumSpatialDims+1, ScalarType>
    differentialVector(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        Plato::Array<mNumSpatialDims+1, ScalarType> tReturnVec;
        tReturnVec(0) = aJacobian(0,0)*aJacobian(0,0);
        tReturnVec(1) = aJacobian(0,1)*aJacobian(0,1);

        return tReturnVec;
    }
};

} // end namespace Plato
