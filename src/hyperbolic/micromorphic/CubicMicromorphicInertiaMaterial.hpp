#pragma once

#include "hyperbolic/micromorphic/MicromorphicInertiaMaterial.hpp"

namespace Plato
{

template<Plato::OrdinalType SpatialDim>
class CubicMicromorphicInertiaMaterial : public MicromorphicInertiaMaterial<SpatialDim>
{
public:
    CubicMicromorphicInertiaMaterial(const Teuchos::ParameterList& aParamList);

    virtual ~CubicMicromorphicInertiaMaterial(){}
};

}
