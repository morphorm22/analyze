/*
 * PlatoProblemFactory.hpp
 *
 *  Created on: Apr 19, 2018
 */

#ifndef PLATOPROBLEMFACTORY_HPP_
#define PLATOPROBLEMFACTORY_HPP_

#include <memory>
#include <sstream>
#include <Teuchos_ParameterList.hpp>

#include "PlatoMesh.hpp"
#include "AnalyzeMacros.hpp"
#include "Mechanics.hpp"
#include "Thermal.hpp"
#include "Tet10.hpp"
#include "Tet4.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"
#include "alg/ParallelComm.hpp"

#ifdef PLATO_HEX_ELEMENTS
#include "Hex8.hpp"
#include "Hex27.hpp"
#include "Quad4.hpp"
#endif

#ifdef PLATO_PLASTICITY
#include "PlasticityProblem.hpp"
#endif

#ifdef PLATO_ELLIPTIC
#include "elliptic/Problem.hpp"
  #ifdef PLATO_HATCHING
  #include "elliptic/hatching/Mechanics.hpp"
  #include "elliptic/hatching/Problem.hpp"
  #endif
#endif

#ifdef PLATO_PARABOLIC
#include "parabolic/Problem.hpp"
#endif

#ifdef PLATO_HYPERBOLIC
#include "hyperbolic/HyperbolicProblem.hpp"
#include "hyperbolic/FluidsQuasiImplicit.hpp"
#endif

#ifdef PLATO_STABILIZED
#include "EllipticVMSProblem.hpp"
#include "StabilizedMechanics.hpp"
#include "StabilizedThermomechanics.hpp"
#endif

#ifdef PLATO_HELMHOLTZ
#include "helmholtz/Helmholtz.hpp"
#include "helmholtz/Problem.hpp"
#endif

//#include "StructuralDynamicsProblem.hpp"

namespace Plato
{

/******************************************************************************//**
* \brief Check if input PDE type is supported by Analyze.
* \param [in] aPlatoProb input xml metadata
* \returns return lowercase pde type
**********************************************************************************/
inline std::string
is_pde_constraint_supported
(Teuchos::ParameterList & aPlatoProb)
{
    if(aPlatoProb.isParameter("PDE Constraint") == false)
    {
        ANALYZE_THROWERR("Parameter 'PDE Constraint' is not defined in 'Plato Problem' parameter list.")
    }
    auto tPDE = aPlatoProb.get < std::string > ("PDE Constraint");
    auto tLowerPDE = Plato::tolower(tPDE);
    return tLowerPDE;
}
// function is_pde_constraint_supported

template<template <typename> typename ProblemT, template <typename> typename PhysicsT>
inline
std::shared_ptr<Plato::AbstractProblem>
makeProblem(
    Plato::Mesh              aMesh,
    Teuchos::ParameterList & aPlatoProb,
    Comm::Machine            aMachine
)
{
    auto tElementType = aMesh->ElementType();
    if( Plato::tolower(tElementType) == "tet10" ||
        Plato::tolower(tElementType) == "tetra10" )
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Tet10>>>(aMesh, aPlatoProb, aMachine);
    }
    else
    if( Plato::tolower(tElementType) == "tetra"  ||
        Plato::tolower(tElementType) == "tetra4" ||
        Plato::tolower(tElementType) == "tet4" )
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Tet4>>>(aMesh, aPlatoProb, aMachine);
    }
    else
    if( Plato::tolower(tElementType) == "hex8" ||
        Plato::tolower(tElementType) == "hexa8" )
    {
#ifdef PLATO_HEX_ELEMENTS
        return std::make_shared<ProblemT<PhysicsT<Plato::Hex8>>>(aMesh, aPlatoProb, aMachine);
#else
        ANALYZE_THROWERR("Not compiled with hex8 elements");
#endif
    }
    else
    if( Plato::tolower(tElementType) == "hex27" ||
        Plato::tolower(tElementType) == "hexa27" )
    {
#ifdef PLATO_HEX_ELEMENTS
        return std::make_shared<ProblemT<PhysicsT<Plato::Hex27>>>(aMesh, aPlatoProb, aMachine);
#else
        ANALYZE_THROWERR("Not compiled with hex27 elements");
#endif
    }
    else
    if( Plato::tolower(tElementType) == "quad4" )
    {
#ifdef PLATO_HEX_ELEMENTS
        return std::make_shared<ProblemT<PhysicsT<Plato::Quad4>>>(aMesh, aPlatoProb, aMachine);
#else
        ANALYZE_THROWERR("Not compiled with quad4 elements");
#endif
    }
    else
    {
        std::stringstream ss;
        ss << "Unknown mesh type: " << tElementType;
        ANALYZE_THROWERR(ss.str());
    }
}


/******************************************************************************//**
* \brief Create mechanical problem.
* \param [in] aMesh        plato abstract mesh
* \param [in] aPlatoProb   input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type mechanical
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_mechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
{
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if (tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Mechanics>(aMesh, aPlatoProb, aMachine);
    }
    else
  #ifdef PLATO_HATCHING
    if(tLowerPDE == "elliptic hatching")
    {
        return makeProblem<Plato::Elliptic::Hatching::Problem, Plato::Elliptic::Hatching::Mechanics>(aMesh, aPlatoProb, aMachine);
    }
  #endif
#endif
#ifdef PLATO_HYPERBOLIC
    else
    if (tLowerPDE == "hyperbolic")
    {
        return std::make_shared<HyperbolicProblem<::Plato::Hyperbolic::Mechanics<SpatialDim>>>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
}
// function create_mechanical_problem

/******************************************************************************//**
* \brief Create plasticity problem.
* \param [in] aMesh        Plato mesh database
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type plasticity
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_plasticity_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
#ifdef PLATO_PLASTICITY
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < PlasticityProblem<::Plato::InfinitesimalStrainPlasticity<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_plasticity_problem

/******************************************************************************//**
* \brief Create a thermoplasticity problem.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type thermoplasticity
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_thermoplasticity_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
#ifdef PLATO_PLASTICITY
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < PlasticityProblem<::Plato::InfinitesimalStrainThermoPlasticity<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_thermoplasticity_problem

/******************************************************************************//**
* \brief Create a abstract problem of type stabilized mechanical.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type stabilized mechanical
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_stabilized_mechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);
#ifdef PLATO_ELLIPTIC
#ifdef PLATO_STABILIZED
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < EllipticVMSProblem<::Plato::StabilizedMechanics<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
}
 // function create_stabilized_mechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type thermal.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type thermal
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_thermal_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_PARABOLIC
    if(tLowerPDE == "parabolic")
    {
        return makeProblem<Plato::Parabolic::Problem, Plato::Thermal>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Thermal>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_thermal_problem

/******************************************************************************//**
* \brief Create a abstract problem of type electromechanical.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type electromechanical
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_electromechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Electromechanics>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_electromechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type stabilized thermomechanical.
* \param [in] aMesh      mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type stabilized thermomechanical
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_stabilized_thermomechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
#ifdef PLATO_STABILIZED
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < EllipticVMSProblem<::Plato::StabilizedThermomechanics<SpatialDim>> > (aMesh, aPlatoProb, aMachine);
        tOutput->readEssentialBoundaryConditions(aPlatoProb);
        return tOutput;
    }
    else
#endif
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_stabilized_thermomechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type thermomechanical.
* \param [in] aMesh        mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type thermomechanical
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_thermomechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_PARABOLIC
    if(tLowerPDE == "parabolic")
    {
        return makeProblem<Plato::Parabolic::Problem, Plato::Thermomechanics>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Thermomechanics>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_thermomechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type incompressible fluid.
* \param [in] aMesh        mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type incompressible fluid
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_incompressible_fluid_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_HYPERBOLIC
    if (tLowerPDE == "hyperbolic")
    {
        return std::make_shared<Plato::Fluids::QuasiImplicit<::Plato::IncompressibleFluids<SpatialDim>>>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_incompressible_fluid_problem

/******************************************************************************//**
* \brief Create a abstract problem of micromorphic mechanics.
* \param [in] aMesh        mesh metadata
* \param [in] aPlatoProb input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type micromorphic mechanics
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline
std::shared_ptr<Plato::AbstractProblem>
create_micromorphic_mechanics_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aPlatoProb,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_HYPERBOLIC
    if (tLowerPDE == "hyperbolic")
    {
        return std::make_shared<HyperbolicProblem<::Plato::Hyperbolic::MicromorphicMechanics<SpatialDim>>>(aMesh, aPlatoProb, aMachine);
    }
    else
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_incompressible_fluid_problem

/******************************************************************************//**
 * \brief This class is responsible for the creation of a Plato problem, which enables
 * finite element simulations of multiphysics problem.
 **********************************************************************************/
class ProblemFactory
{
public:
    /******************************************************************************//**
     * \brief Returns a shared pointer to a PLATO problem
     * \param [in] aMesh        abstract mesh
     * \param [in] aInputParams xml metadata
     * \param [in] aMachine     mpi communicator interface
     * \returns shared pointer to a PLATO problem
     **********************************************************************************/
    std::shared_ptr<Plato::AbstractProblem>
    create(
        Plato::Mesh              aMesh,
        Teuchos::ParameterList & aInputParams,
        Comm::Machine            aMachine
    )
    {
        auto tInputData = aInputParams.sublist("Plato Problem");
        auto tPhysics = tInputData.get < std::string > ("Physics");
        auto tLowerPhysics = Plato::tolower(tPhysics);

        if(tLowerPhysics == "mechanical")
        {
            return ( Plato::create_mechanical_problem(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "plasticity")
        {
            return ( Plato::create_plasticity_problem(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "thermoplasticity")
        {
            return ( Plato::create_thermoplasticity_problem(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "stabilized mechanical")
        {
            return ( Plato::create_stabilized_mechanical_problem(aMesh, tInputData, aMachine) );
        }
        else if(tLowerPhysics == "thermal")
        {
            return ( Plato::create_thermal_problem(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "electromechanical")
        {
            return ( Plato::create_electromechanical_problem(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "stabilized thermomechanical")
        {
            return ( Plato::create_stabilized_thermomechanical_problem(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "thermomechanical")
        {
            return ( Plato::create_thermomechanical_problem(aMesh, tInputData, aMachine) );
        }
#ifdef PLATO_HYPERBOLIC
        else
        if(tLowerPhysics == "incompressible fluids")
        {
            return ( Plato::create_incompressible_fluid_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
        else
        if(tLowerPhysics == "micromorphic mechanical")
        {
            return ( Plato::create_micromorphic_mechanics_problem<SpatialDim>(aMesh, tInputData, aMachine) );
        }
#endif
#ifdef PLATO_HELMHOLTZ
        else
        if(tLowerPhysics == "helmholtz filter")
        {
            return makeProblem<Plato::Helmholtz::Problem, Plato::HelmholtzFilter>(aMesh, tInputData, aMachine);
        }
#endif
        else
        {
            ANALYZE_THROWERR(std::string("'Physics' of type ") + tLowerPhysics + "' is not supported.");
        }
        return nullptr;
    }
};
// class ProblemFactory

}
// namespace Plato

#endif /* PLATOPROBLEMFACTORY_HPP_ */
