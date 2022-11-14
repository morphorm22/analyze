#ifndef PLATO_HELMHOLTZ_HPP
#define PLATO_HELMHOLTZ_HPP

#include "PlatoUtilities.hpp"
#include "helmholtz/AbstractVectorFunction.hpp"
#include "helmholtz/HelmholtzResidual.hpp"

namespace Plato {

namespace HelmholtzFactory {
/******************************************************************************/
struct FunctionFactory{
/******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Helmholtz::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              aFunctionType
    )
    {

        auto tFunctionType = Plato::tolower(aFunctionType);
        if( tFunctionType == "helmholtz filter" )
        {
            return std::make_shared<Plato::Helmholtz::HelmholtzResidual<EvaluationType>>
              (aSpatialDomain, aDataMap, aParamList);
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList for HELMHOLTZ FILTER");
        }
    }

};

} // namespace HelmholtzFactory

} // namespace Plato

#include "HelmholtzElement.hpp"

namespace Plato {

template <typename TopoElementType>
class HelmholtzFilter
{
public:
    typedef Plato::HelmholtzFactory::FunctionFactory FunctionFactory;
    using ElementType = HelmholtzElement<TopoElementType>;
};
// class HelmholtzFilter

} //namespace Plato

#endif
