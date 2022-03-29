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

// TODO #include "Thermal.hpp"
#include "Mechanics.hpp"
// TODO #include "Electromechanics.hpp"
// TODO #include "Thermomechanics.hpp"

#ifdef PLATO_STABILIZED
// TODO #include "StabilizedMechanics.hpp"
// TODO #include "StabilizedThermomechanics.hpp"
#endif

#ifdef PLATOANALYZE_1D
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermal<1>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Mechanics<1>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<1>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<1>>;
#ifdef PLATO_STABILIZED
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<1>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<1>>;
#endif
#endif

#ifdef PLATOANALYZE_2D
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermal<2>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Mechanics<2>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<2>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<2>>;
#ifdef PLATO_STABILIZED
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<2>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<2>>;
#endif
#endif

#ifdef PLATOANALYZE_3D
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermal<3>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Mechanics<3>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<3>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<3>>;
#ifdef PLATO_STABILIZED
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<3>>;
// TODO extern template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<3>>;
#endif
#endif
