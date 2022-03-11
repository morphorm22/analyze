/*
 * MicromorphicInertiaModelFactory.hpp
 *
 *  Created on: Oct 27, 2021
 */

#pragma once

#include "hyperbolic/MicromorphicInertiaMaterial.hpp"

#include "hyperbolic/CubicMicromorphicInertiaMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Factory for creating micromorphic inertia models.
 *
 * \tparam SpatialDim spatial dimensions: options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class MicromorphicInertiaModelFactory
{
public:
    /******************************************************************************//**
    * \brief Micromorpic inertia model factory constructor.
    * \param [in] aParamList input parameter list
    **********************************************************************************/
    MicromorphicInertiaModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList){}

    /******************************************************************************//**
    * \brief Create a micromorphic inertia model.
    * \param [in] aModelName name of the model to be created.
    * \return Teuchos reference counter pointer to inertia model
    **********************************************************************************/
    Teuchos::RCP<Plato::MicromorphicInertiaMaterial<SpatialDim>>
    create(std::string aModelName)
    {
        if (!mParamList.isSublist("Material Models"))
        {
            REPORT("'Material Models' list not found! Returning 'nullptr'");
            return Teuchos::RCP<Plato::MicromorphicInertiaMaterial<SpatialDim>>(nullptr);
        }
        else
        {
            auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");
           
            if (!tModelsParamList.isSublist(aModelName))
            {
                std::stringstream ss;
                ss << "Requested an inertia material model ('" << aModelName << "') that isn't defined";
                ANALYZE_THROWERR(ss.str());
            }

            auto tModelParamList = tModelsParamList.sublist(aModelName);
            if(tModelParamList.isSublist("Cubic Micromorphic Inertia"))
            {
                return Teuchos::rcp(new Plato::CubicMicromorphicInertiaMaterial<SpatialDim>(tModelParamList.sublist("Cubic Micromorphic Inertia")));
            }
            return Teuchos::RCP<Plato::MicromorphicInertiaMaterial<SpatialDim>>(nullptr);
        }
    }

private:
    const Teuchos::ParameterList& mParamList; /*!< Input parameter list */
};
// class MicromorphicInertiaModelFactory

}
// namespace Plato
