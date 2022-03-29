/*
 * SurfaceIntegralUtilities.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "alg/PlatoLambda.hpp"
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
class SurfaceArea
{
    using Body = ElementType;
    using Face = typename ElementType::Face;

public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    SurfaceArea(){}

    /***************************************************************************//**
     * \brief Calculate surface area.
     *
     * \tparam ConfigScalarType configuration forward automatically differentiation (FAD) type
     * \tparam ResultScalarType output FAD type
     *
     * \param [in]  aCellOrdinal  cell ordinal
     * \param [in]  aPointOrdinal cubature point ordinal
     * \param [in]  aBasisValues  basis function values
     * \param [in]  aConfig       cell/element node coordinates
     * \param [out] aOutput       surface area container
     *
    *******************************************************************************/
    template<typename ConfigScalarType, typename ResultScalarType>
    DEVICE_TYPE inline void
    operator()(
        const Plato::OrdinalType                         & aCellOrdinal,
        const Plato::Array<Face::mNumNodesPerCell,
                           Plato::OrdinalType>           & aLocalNodeOrds,
        const Plato::Matrix<Face::mNumNodesPerCell,
                            Face::mNumSpatialDims>       & aBasisGrads,
        const Plato::ScalarArray3DT<ConfigScalarType>    & aConfig,
              ResultScalarType                           & aResult
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
        aResult = Face::differentialMeasure(tJacobian);
    }
};
// class SurfaceArea

}
// namespace Plato
