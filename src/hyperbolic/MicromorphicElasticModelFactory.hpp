/*
 * MicromorphicElasticModelFactory.hpp
 *
 *  Created on: Oct 19, 2021
 */

#pragma once

#include "hyperbolic/MicromorphicLinearElasticMaterial.hpp"

#include "hyperbolic/CubicMicromorphicLinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Factory for creating linear elastic material models.
 *
 * \tparam SpatialDim spatial dimensions: options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class MicromorphicElasticModelFactory
{
public:
    /******************************************************************************//**
    * \brief Micromorpic linear elastic material model factory constructor.
    * \param [in] aParamList input parameter list
    **********************************************************************************/
    MicromorphicElasticModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList){}

    /******************************************************************************//**
    * \brief Create a micromorphic linear elastic material model.
    * \param [in] aModelName name of the model to be created.
    * \return Teuchos reference counter pointer to linear elastic material model
    **********************************************************************************/
    Teuchos::RCP<Plato::MicromorphicLinearElasticMaterial<SpatialDim>>
    create(std::string aModelName)
    {
        if (!mParamList.isSublist("Material Models"))
        {
            REPORT("'Material Models' list not found! Returning 'nullptr'");
            return Teuchos::RCP<Plato::MicromorphicLinearElasticMaterial<SpatialDim>>(nullptr);
        }
        else
        {
            auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");
           
            if (!tModelsParamList.isSublist(aModelName))
            {
                std::stringstream ss;
                ss << "Requested a material model ('" << aModelName << "') that isn't defined";
                THROWERR(ss.str());
            }

            auto tModelParamList = tModelsParamList.sublist(aModelName);
            if(tModelParamList.isSublist("Cubic Micromorphic Linear Elastic"))
            {
                return Teuchos::rcp(new Plato::CubicMicromorphicLinearElasticMaterial<SpatialDim>(tModelParamList.sublist("Cubic Micromorphic Linear Elastic")));
            }
            return Teuchos::RCP<Plato::MicromorphicLinearElasticMaterial<SpatialDim>>(nullptr);
        }
    }

private:
    const Teuchos::ParameterList& mParamList; /*!< Input parameter list */
};
// class MicromorphicElasticModelFactory

}
// namespace Plato
