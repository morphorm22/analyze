#pragma once

#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! This element base provides basic functions
*/
/******************************************************************************/
template <typename ElementType>
class ElementBase
{

    static constexpr Plato::OrdinalType mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumNodesPerCell = ElementType::mNumNodesPerCell;
  public:

    template<typename ScalarType>
    DEVICE_TYPE static inline Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ScalarType>
    jacobian(
        const Plato::Array<mNumSpatialDims>&    aCubPoint,
              Plato::ScalarArray3DT<ScalarType> aConfig,
              Plato::OrdinalType                aCellOrdinal
    )
    {
        Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ScalarType> tJacobian;
        auto tBasisGrads = ElementType::basisGrads(aCubPoint);
        for (int i=0; i<mNumSpatialDims; i++)
        {
            for (int j=0; j<mNumSpatialDims; j++)
            {
                tJacobian(i,j) = ScalarType(0.0);
                for (int I=0; I<mNumNodesPerCell; I++)
                {
                    tJacobian(i,j) += tBasisGrads(I,j)*aConfig(aCellOrdinal,I,i);
                }
            }
        }
        return tJacobian;
    }
    template<typename ScalarType>
    DEVICE_TYPE static inline Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ScalarType>
    jacobian(
        const Plato::Array<mNumSpatialDims>         & aCubPoint,
              Plato::ScalarMultiVectorT<ScalarType>   aConfig
    )
    {
        Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ScalarType> tJacobian;
        auto tBasisGrads = ElementType::basisGrads(aCubPoint);
        for (int i=0; i<mNumSpatialDims; i++)
        {
            for (int j=0; j<mNumSpatialDims; j++)
            {
                tJacobian(i,j) = ScalarType(0.0);
                for (int I=0; I<mNumNodesPerCell; I++)
                {
                    tJacobian(i,j) += tBasisGrads(I,j)*aConfig(I,i);
                }
            }
        }
        return tJacobian;
    }

    template<typename ScalarType>
    DEVICE_TYPE static inline void
    computeGradientMatrix(
        const Plato::Array<mNumSpatialDims>                                & aCubPoint,
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ScalarType>  & aJacInv,
              Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ScalarType> & aGradient
    )
    {
        auto tBasisGrads = ElementType::basisGrads(aCubPoint);
        for (int I=0; I<mNumNodesPerCell; I++)
        {
            for (int k=0; k<mNumSpatialDims; k++)
            {
                aGradient(I, k) = ScalarType(0.0);
                for (int j=0; j<mNumSpatialDims; j++)
                {
                    aGradient(I, k) += tBasisGrads(I,j)*aJacInv(j,k);
                }
            }
        }
    }
};

} // end namespace Plato
