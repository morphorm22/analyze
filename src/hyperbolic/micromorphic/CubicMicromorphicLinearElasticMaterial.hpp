/*
 * CubicMicromorphicLinearElasticMaterial.hpp
 *
 *  Created on: Oct 18, 2021
 */

#pragma once

#include "hyperbolic/micromorphic/MicromorphicLinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for Cubic linear elastic micromorphic material models
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class CubicMicromorphicLinearElasticMaterial : public MicromorphicLinearElasticMaterial<SpatialDim>
{
public:
    /******************************************************************************//**
     * \brief Linear elastic Cubic micromorphic material model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    CubicMicromorphicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Linear elastic Cubic micromorphic material model destructor.
    **********************************************************************************/
    virtual ~CubicMicromorphicLinearElasticMaterial(){}
};
// class CubicMicromorphicLinearElasticMaterial

}
// namespace Plato
