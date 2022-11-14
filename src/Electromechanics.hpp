#ifndef PLATO_ELECTROMECHANICS_HPP
#define PLATO_ELECTROMECHANICS_HPP

#include <memory>

#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/ElectroelastostaticResidual.hpp"
#include "elliptic/InternalElectroelasticEnergy.hpp"
#include "elliptic/EMStressPNorm.hpp"

#include "MakeFunctions.hpp"

namespace Plato
{

namespace ElectromechanicsFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aParamList,
              std::string              aFuncType
    )
    /******************************************************************************/
    {

        auto tLowerFuncType = Plato::tolower(aFuncType);
        if(tLowerFuncType == "elliptic")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::ElectroelastostaticResidual>
                     (aSpatialDomain, aDataMap, aParamList, aFuncType);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              std::string              aFuncType,
              std::string              aFuncName
    )
    /******************************************************************************/
    {
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if(tLowerFuncType == "internal electroelastic energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::InternalElectroelasticEnergy>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        if(tLowerFuncType == "stress p-norm")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::EMStressPNorm>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
            throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }
}; // struct FunctionFactory

} // namespace ElectromechanicsFactory

} // namespace Plato

#include "ElectromechanicsElement.hpp"

namespace Plato
{
template<typename TopoElementType>
class Electromechanics
{
public:
    typedef Plato::ElectromechanicsFactory::FunctionFactory FunctionFactory;
    using ElementType = ElectromechanicsElement<TopoElementType>;
};

} // namespace Plato

#endif
