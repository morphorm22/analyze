#pragma once

#include <memory>

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Scalar function base factory
 **********************************************************************************/
template<typename PhysicsType>
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
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> 
    create(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aFunctionName);
};
// class ScalarFunctionBaseFactory

} // namespace Elliptic

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEC(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Thermal)
PLATO_ELEMENT_DEC(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Mechanics)
PLATO_ELEMENT_DEC(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Thermomechanics)
PLATO_ELEMENT_DEC(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
#include "StabilizedMechanics.hpp"
#include "StabilizedThermomechanics.hpp"

PLATO_ELEMENT_DEC(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::StabilizedMechanics)
PLATO_ELEMENT_DEC(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::StabilizedThermomechanics)
#endif
