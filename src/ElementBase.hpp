#pragma once

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! This element base provides basic functions
*/
/******************************************************************************/
template <typename ElementType>
class ElementBase
{

  public:
    static constexpr Plato::OrdinalType mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumNodesPerCell = ElementType::mNumNodesPerCell;

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ScalarType>
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
    KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ScalarType>
    jacobian(
      const Plato::Array<mNumSpatialDims>   & aCubPoint,
            Plato::Matrix<mNumNodesPerCell,
                          mNumSpatialDims,
                          ScalarType>       & aNodeLocations
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
                    tJacobian(i,j) += tBasisGrads(I,j)*aNodeLocations(I,i);
                }
            }
        }
        return tJacobian;
    }

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION static void
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
