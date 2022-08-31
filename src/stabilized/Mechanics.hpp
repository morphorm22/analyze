#pragma once

#include <memory>

#include "elliptic/AbstractScalarFunction.hpp"
#include "stabilized/Projection.hpp"
#include "stabilized/ElastostaticResidual.hpp"
#include "stabilized/ElastostaticEnergy.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Stabilized
{

namespace MechanicsFactory
{

/******************************************************************************//**
 * \brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO vector function (i.e. residual equation)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Analyze physics-based database
     * \param [in] aProblemParams input parameters
     * \param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Stabilized::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aPDE
    )
    {
        auto tLowerPDE = Plato::tolower(aPDE);

        if(tLowerPDE == "elliptic")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Stabilized::ElastostaticResidual>
                     (aSpatialDomain, aDataMap, aProblemParams, aPDE);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aFuncType,
        const std::string            & aFuncName
    )
    /******************************************************************************/
    {
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if(tLowerFuncType == "internal elastic energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Stabilized::ElastostaticEnergy>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
            ANALYZE_THROWERR("Unknown scalar function specified in 'Plato Problem' ParameterList");
        }
    }
};
// struct FunctionFactory

} // namespace MechanicsFactory
} // namespace Stabilized
} // namespace Plato

#include "stabilized/MechanicsElement.hpp"

namespace Plato {
namespace Stabilized {
/****************************************************************************//**
 * \brief Concrete class for use as the PhysicsType template argument in Problem
 *******************************************************************************/
template<typename TopoElementType>
class Mechanics
{
public:
    typedef Plato::Stabilized::MechanicsFactory::FunctionFactory FunctionFactory;

    using ElementType   = Plato::Stabilized::MechanicsElement<TopoElementType>;
    using ProjectorType = typename Plato::Stabilized::Projection<TopoElementType,
                                                                 ElementType::mNumDofsPerNode,
                                                                 ElementType::mPressureDofOffset,
                                                                 /* numProjectionDofs=*/ 1>;
};

} // namespace Stabilized
} // namespace Plato
