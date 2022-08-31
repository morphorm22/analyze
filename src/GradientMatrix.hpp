#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato {

/******************************************************************************//**
 * \brief Gradient matrix functor
**********************************************************************************/
template<typename ElementType>
class ComputeGradientMatrix : public ElementType
{
  public:

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
              Plato::OrdinalType aCellOrdinal,
        const Plato::Array<ElementType::mNumSpatialDims> & aCubPoint,
              Plato::ScalarArray3DT<ScalarType>    aConfig,
              Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ScalarType> & aGradient,
              ScalarType&     aVolume
    ) const
    {
        auto tJacobian = ElementType::jacobian(aCubPoint, aConfig, aCellOrdinal);
        aVolume = Plato::determinant(tJacobian);
        auto tJacInv = Plato::invert(tJacobian);
        ElementType::template computeGradientMatrix<ScalarType>(aCubPoint, tJacInv, aGradient);
    }

    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
              Plato::OrdinalType aCellOrdinal,
              Plato::OrdinalType aGpOrdinal,
        const Plato::Array<ElementType::mNumSpatialDims> & aCubPoint,
              Plato::ScalarArray3DT<ScalarType>            aConfig,
              Plato::ScalarArray4DT<ScalarType>            aGradient,
              Plato::ScalarMultiVectorT<ScalarType>        aVolume
    ) const
    {
        auto tJacobian = ElementType::jacobian(aCubPoint, aConfig, aCellOrdinal);
        aVolume(aCellOrdinal, aGpOrdinal) = Plato::determinant(tJacobian);
        auto tJacInv = Plato::invert(tJacobian);

        Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ScalarType> tGradient;
        ElementType::template computeGradientMatrix<ScalarType>(aCubPoint, tJacInv, tGradient);

        for (int I=0; I<ElementType::mNumNodesPerCell; I++)
        {
            for (int k=0; k<ElementType::mNumSpatialDims; k++)
            {
                aGradient(aCellOrdinal, aGpOrdinal, I, k) = tGradient(I, k);
            }
        }
    }
};

}
