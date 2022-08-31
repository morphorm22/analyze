#pragma once

#include "hyperbolic/micromorphic/MicromorphicLinearElasticMaterial.hpp"

namespace Plato
{

template<Plato::OrdinalType SpatialDim>
class CubicMicromorphicLinearElasticMaterial : public MicromorphicLinearElasticMaterial<SpatialDim>
{
public:
    CubicMicromorphicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    virtual ~CubicMicromorphicLinearElasticMaterial(){}
};

}
