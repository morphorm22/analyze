/*
 * FluidsVolumeIntegrandFactory.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "SpatialModel.hpp"
#include "PlatoUtilities.hpp"

#include "hyperbolic/fluids/AbstractVolumeIntegrand.hpp"
#include "hyperbolic/fluids/FluidsUtils.hpp"
#include "hyperbolic/fluids/InternalThermalForces.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \strut VolumeIntegrandFactory
 *
 * \brief Factory for internal force integrals for computational fluid dynamics
 *   applications.
 *
 ******************************************************************************/
struct VolumeIntegrandFactory
{

/***************************************************************************//**
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \fn inline std::shared_ptr<AbstractVolumeIntegrand> createInternalThermalForces
 *
 * \brief Return shared pointer to an abstract cell volume integral instance.
 *
 * \param [in] aDomain  spatial domain metadata
 * \param [in] aDataMap output database
 * \param [in] aInputs  input file metadata
 *
 ******************************************************************************/
template <typename PhysicsT, typename EvaluationT>
inline std::shared_ptr<Plato::AbstractVolumeIntegrand<PhysicsT, EvaluationT>>
createInternalThermalForces
(const Plato::SpatialDomain & aDomain,
 Plato::DataMap & aDataMap,
 Teuchos::ParameterList & aInputs)
{
    auto tScenario = Plato::Fluids::scenario(aInputs);
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if( tScenario == "density-based topology optimization" && tHeatTransfer == "forced")
    {
        return ( std::make_shared<Plato::Fluids::SIMP::InternalThermalForces<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else if( tScenario == "density-based topology optimization" && tHeatTransfer == "natural")
    {
        return ( std::make_shared<Plato::Fluids::InternalThermalForces<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else if( tScenario == "analysis" || tScenario == "levelset topology optimization" )
    {
        return ( std::make_shared<Plato::Fluids::InternalThermalForces<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else
    {
        ANALYZE_THROWERR( std::string("Requested Use Case: Scenario '") + tScenario + "' & Heat Transfer Mechanism '" + tHeatTransfer + "' is not supported. " +
            "Supported use cases are: Scenario = 'Density-Based Topology Optimization' and Heat Transfer = 'Natural or Forced', " + 
            "Scenario = 'Levelset Topology Optimization' & Heat Transfer = 'Natural or Forced', " + 
            "Scenario = 'Analysis' & Heat Transfer = 'Natural or Forced'" )
    }
}

};
// struct VolumeIntegrandFactory

}
// namespace Fluids

}
// namespace Plato
