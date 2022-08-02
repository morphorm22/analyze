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
    DEVICE_TYPE inline void
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

};

}
