#pragma once

#include "hyperbolic/micromorphic/MicromorphicInertiaMaterial.hpp"

#include "hyperbolic/micromorphic/CubicMicromorphicInertiaMaterial.hpp"

namespace Plato
{

template<Plato::OrdinalType SpatialDim>
class MicromorphicInertiaModelFactory
{
public:
    MicromorphicInertiaModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList){}

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
    const Teuchos::ParameterList& mParamList; 
};

}
