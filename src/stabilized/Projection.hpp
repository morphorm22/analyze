#pragma once

#include <memory>

#include "ProjectionElement.hpp"
#include "PressureGradientProjectionResidual.hpp"
#include "MakeFunctions.hpp"

namespace Plato
{

namespace Stabilized
{

namespace ProjectionFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Stabilized::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aFunctionType
    )
    /******************************************************************************/
    {
        if(aFunctionType == "State Gradient Projection")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Stabilized::PressureGradientProjectionResidual>
                     (aSpatialDomain, aDataMap, aProblemParams, aFunctionType);
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
}; // struct FunctionFactory

} // namespace ProjectionFactory


/****************************************************************************//**
 * \brief Concrete class for use as the PhysicsT template argument in VectorFunction
 *******************************************************************************/
template<
  typename TopoElementType,
  Plato::OrdinalType TotalDofsParam = TopoElementType::mNumSpatialDims,
  Plato::OrdinalType ProjectionDofOffset = 0,
  Plato::OrdinalType NumProjectionDofs = 1>
class Projection
{
public:
    typedef Plato::Stabilized::ProjectionFactory::FunctionFactory FunctionFactory;
    using ElementType = Plato::Stabilized::ProjectionElement<TopoElementType, TotalDofsParam, ProjectionDofOffset, NumProjectionDofs>;
};

} // namespace Stabilized
} // namespace Plato
