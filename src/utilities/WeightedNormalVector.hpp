/*
 * WeightedNormalVector.hpp
 *
 *  Created on: Apr 19, 2022
 */

#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Compute cubature weight for surface integrals.
 *
 * \tparam ElementType Element type
 *
*******************************************************************************/
template<typename ElementType>
class WeightedNormalVector
{
    using Body = ElementType;
    using Face = typename ElementType::Face;

public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    WeightedNormalVector(){}

    /***************************************************************************//**
     * \brief Calculate surface area weighted normal vector.
     *
     * \tparam ConfigScalarType configuration forward automatically differentiation (FAD) type
     * \tparam ResultScalarType output FAD type
     *
     * \param [in]  aCellOrdinal  cell ordinal
     * \param [in]  aPointOrdinal cubature point ordinal
     * \param [in]  aBasisValues  basis function values
     * \param [in]  aConfig       cell/element node coordinates
     * \param [out] aResult       surface area weighted normal
     *
    *******************************************************************************/
    template<typename ConfigScalarType, typename ResultScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        const Plato::OrdinalType                         & aCellOrdinal,
        const Plato::Array<Face::mNumNodesPerCell,
                           Plato::OrdinalType>           & aLocalNodeOrds,
        const Plato::Matrix<Face::mNumNodesPerCell,
                            Face::mNumSpatialDims>       & aBasisGrads,
        const Plato::ScalarArray3DT<ConfigScalarType>    & aConfig,
              Plato::Array<Body::mNumSpatialDims,
                           ResultScalarType>             & aResult
    ) const
    {
        Plato::Matrix<Face::mNumSpatialDims, Body::mNumSpatialDims, ConfigScalarType> tJacobian(0.0);

        for(Plato::OrdinalType iFace=0; iFace<Face::mNumSpatialDims; iFace++)
        {
            for(Plato::OrdinalType iBody=0; iBody<Body::mNumSpatialDims; iBody++)
            {
                for(Plato::OrdinalType iNode=0; iNode<Face::mNumNodesPerCell; iNode++)
                {
                    tJacobian(iFace, iBody) += aBasisGrads(iNode,iFace)*aConfig(aCellOrdinal,aLocalNodeOrds(iNode),iBody);
                }
            }
        }
        aResult = Face::differentialVector(tJacobian);
    }
};
// class WeightedNormalVector

}
// namespace Plato
