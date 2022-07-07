#pragma once

#include <memory>

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "parabolic/ScalarFunctionBase.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Parabolic
{
/******************************************************************************//**
 * \brief Scalar function base factory
 **********************************************************************************/
template<typename PhysicsT>
class ScalarFunctionBaseFactory
{
public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    ScalarFunctionBaseFactory () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~ScalarFunctionBaseFactory() {}

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::Parabolic::ScalarFunctionBase> 
    create(
        Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
        std::string            & aFunctionName);
};
// class ScalarFunctionBaseFactory

} // namespace Parabolic

} // namespace Plato

#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEC(Plato::Parabolic::ScalarFunctionBaseFactory, Plato::Thermal)
PLATO_ELEMENT_DEC(Plato::Parabolic::ScalarFunctionBaseFactory, Plato::Thermomechanics)
