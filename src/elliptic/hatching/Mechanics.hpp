#pragma once

#include <memory>

#include "elliptic/hatching/AbstractScalarFunction.hpp"
#include "elliptic/hatching/InternalElasticEnergy.hpp"
#include "elliptic/hatching/ElastostaticResidual.hpp"

// TODO
// #include "elliptic/StressPNorm.hpp"

#include "MakeFunctions.hpp"

#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

namespace MechanicsFactory
{

/******************************************************************************//**
 * @brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO vector function (i.e. residual equation)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze physics-based database
     * \param [in] aProblemParams input parameters
     * \param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::Hatching::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              std::string              aPDE)
    {
        auto tLowerPDE = Plato::tolower(aPDE);

        if(tLowerPDE == "elliptic hatching")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::Hatching::ElastostaticResidual>
                     (aSpatialDomain, aDataMap, aProblemParams, aPDE);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList")
        }
    }


    /******************************************************************************//**
     * \brief Create a PLATO scalar function (i.e. optimization criterion)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze physics-based database
     * \param [in] aProblemParams input parameters
     * \param [in] aFuncType scalar function type
     * \param [in] aFuncName scalar function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::Hatching::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              std::string              aFuncType,
              std::string              aFuncName
    )
    {
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if(tLowerFuncType == "internal elastic energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::Hatching::InternalElasticEnergy>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
// TODO
#ifdef COMING_SOON
        else if(tLowerFuncType == "Stress P-Norm")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::Hatching::StressPNorm>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
#endif
        else
        {
            return nullptr;
        }
    }
};
// struct FunctionFactory

} // namespace MechanicsFactory
} // namespace Hatching
} // namespace Elliptic
} // namespace Plato

#include "elliptic/hatching/MechanicsElement.hpp"

namespace Plato {
namespace Elliptic {
namespace Hatching {
/******************************************************************************//**
 * \brief Concrete class for use as the Physics template argument in
 *        Plato Problem
**********************************************************************************/
template<typename TopoElementType>
class Mechanics
{
public:
    typedef Plato::Elliptic::Hatching::MechanicsFactory::FunctionFactory FunctionFactory;
    using ElementType = Plato::Elliptic::Hatching::MechanicsElement<TopoElementType>;
};

} // namespace Hatching
} // namespace Elliptic
} // namespace Plato
