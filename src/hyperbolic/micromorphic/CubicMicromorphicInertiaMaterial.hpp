/*
 * CubicMicromorphicInertiaMaterial.hpp
 *
 *  Created on: Oct 27, 2021
 */

#pragma once

#include "hyperbolic/micromorphic/MicromorphicInertiaMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for Cubic micromorphic inertia models
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class CubicMicromorphicInertiaMaterial : public MicromorphicInertiaMaterial<SpatialDim>
{
public:
    /******************************************************************************//**
     * \brief Cubic micromorphic inertia model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    CubicMicromorphicInertiaMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Cubic micromorphic inertia model destructor.
    **********************************************************************************/
    virtual ~CubicMicromorphicInertiaMaterial(){}
};
// class CubicMicromorphicInertiaMaterial

}
// namespace Plato
