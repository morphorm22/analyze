#pragma once

#include "hyperbolic/micromorphic/MicromorphicLinearElasticMaterial.hpp"

#include "hyperbolic/micromorphic/CubicMicromorphicLinearElasticMaterial.hpp"

namespace Plato
{

template<Plato::OrdinalType SpatialDim>
class MicromorphicElasticModelFactory
{
public:
    MicromorphicElasticModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList){}

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
                ANALYZE_THROWERR(ss.str());
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
    const Teuchos::ParameterList& mParamList; 
};

}
