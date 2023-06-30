/*
 * PlatoProblemFactory.hpp
 *
 *  Created on: Apr 19, 2018
 */

#pragma once

#include <memory>
#include <sstream>
#include <Teuchos_ParameterList.hpp>
#include <stdexcept>

#include "Tet4.hpp"
#include "Tet10.hpp"
#include "PlatoMesh.hpp"
#include "AnalyzeMacros.hpp"
#include "solver/ParallelComm.hpp"
#include "base/SupportedParamOptions.hpp"

#ifdef PLATO_HEX_ELEMENTS
  #include "Hex8.hpp"
  #include "Hex27.hpp"
  #include "Quad4.hpp"
#endif

#ifdef PLATO_PLASTICITY
  #include "PlasticityProblem.hpp"
#endif

#ifdef PLATO_ELLIPTIC
  #include "elliptic/thermal/Thermal.hpp"
  #include "elliptic/electrical/Electrical.hpp"
  #include "elliptic/mechanical/linear/Mechanics.hpp"
  #include "elliptic/mechanical/nonlinear/Mechanics.hpp"
  #include "elliptic/electromechanics/Electromechanics.hpp"
  #include "elliptic/thermomechanics/linear/Thermomechanics.hpp"
  #include "elliptic/thermomechanics/nonlinear/ThermoMechanics.hpp"
  #include "elliptic/Problem.hpp"
#endif

#ifdef PLATO_PARABOLIC
  #include "parabolic/Thermal.hpp"
  #include "parabolic/Mechanics.hpp"
  #include "parabolic/Thermomechanics.hpp"
  #include "parabolic/Problem.hpp"
#endif

#ifdef PLATO_HYPERBOLIC
  #include "hyperbolic/Problem.hpp"
  #include "hyperbolic/Mechanics.hpp"
    #ifdef PLATO_FLUIDS
      #include "hyperbolic/fluids/FluidsQuasiImplicit.hpp"
    #endif
#endif

#ifdef PLATO_HELMHOLTZ
  #include "helmholtz/Helmholtz.hpp"
  #include "helmholtz/Problem.hpp"
#endif

namespace Plato
{

/******************************************************************************//**
* \brief Check if input PDE type is supported by Analyze.
* \param [in] aProbParams input xml metadata
* \returns return lowercase pde type
**********************************************************************************/
inline 
std::string
get_pde_type(
  Teuchos::ParameterList & aProbParams
)
{
  if(aProbParams.isParameter("PDE Constraint") == false){
    ANALYZE_THROWERR("ERROR: Argument ('PDE Constraint') is not defined in ('Plato Problem') parameter list")
  }
  auto tPDE = aProbParams.get < std::string > ("PDE Constraint");
  auto tLowerPDE = Plato::tolower(tPDE);
  return tLowerPDE;
}
// function get_pde_type

template<template <typename> typename ProblemT, template <typename> typename PhysicsT>
inline
std::shared_ptr<Plato::AbstractProblem>
makeProblem(
    Plato::Mesh              aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine            aMachine
)
{
    auto tElementType = aMesh->ElementType();
    if( Plato::tolower(tElementType) == "tet10" ||
        Plato::tolower(tElementType) == "tetra10" )
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Tet10>>>(aMesh, aProbParams, aMachine);
    }
    if( Plato::tolower(tElementType) == "tetra"  ||
        Plato::tolower(tElementType) == "tetra4" ||
        Plato::tolower(tElementType) == "tet4" )
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Tet4>>>(aMesh, aProbParams, aMachine);
    }
    if( Plato::tolower(tElementType) == "tri"  ||
        Plato::tolower(tElementType) == "tri3" )
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Tri3>>>(aMesh, aProbParams, aMachine);
    }
    if( Plato::tolower(tElementType) == "hex8" ||
        Plato::tolower(tElementType) == "hexa8" )
    {
#ifdef PLATO_HEX_ELEMENTS
        return std::make_shared<ProblemT<PhysicsT<Plato::Hex8>>>(aMesh, aProbParams, aMachine);
#else
        ANALYZE_THROWERR("Not compiled with hex8 elements");
#endif
    }
    if( Plato::tolower(tElementType) == "hex27" ||
        Plato::tolower(tElementType) == "hexa27" )
    {
#ifdef PLATO_HEX_ELEMENTS
        return std::make_shared<ProblemT<PhysicsT<Plato::Hex27>>>(aMesh, aProbParams, aMachine);
#else
        ANALYZE_THROWERR("Not compiled with hex27 elements");
#endif
    }
    if( Plato::tolower(tElementType) == "quad4" )
    {
#ifdef PLATO_HEX_ELEMENTS
        return std::make_shared<ProblemT<PhysicsT<Plato::Quad4>>>(aMesh, aProbParams, aMachine);
#else
        ANALYZE_THROWERR("Not compiled with quad4 elements");
#endif
    }
    {
        std::stringstream ss;
        ss << "Unknown mesh type: " << tElementType;
        ANALYZE_THROWERR(ss.str());
    }
}

/******************************************************************************//**
* \brief Create mechanical problem.
* \param [in] aMesh        plato abstract mesh
* \param [in] aProbParams   input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type mechanical
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_linear_mechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aProbParams,
 Comm::Machine            aMachine)
{
  auto tLowerPDE = Plato::get_pde_type(aProbParams);
#ifdef PLATO_ELLIPTIC
  if (tLowerPDE == "elliptic")
  {
    return makeProblem<Plato::Elliptic::Problem,Plato::Elliptic::Linear::Mechanics>(aMesh, aProbParams, aMachine);
  }
#endif
#ifdef PLATO_HYPERBOLIC
  if (tLowerPDE == "hyperbolic")
  {
    return makeProblem<Plato::Hyperbolic::Problem,Plato::Hyperbolic::Mechanics>(aMesh, aProbParams, aMachine);
  }
#endif
  {
    ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
  }
}
// function create_linear_mechanical_problem

/// @fn create_nonlinear_mechanical_problem
/// @brief inline function, create nonlinear mechanical residual evaluator
/// @param [in] aMesh      contains mesh and model information
/// @param [in] aProbParams inpur problem parameters
/// @param [in] aMachine   contains mpi communicator information
/// @return shared pointer to residual evaluator
inline
std::shared_ptr<Plato::AbstractProblem>
create_nonlinear_mechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aProbParams,
 Comm::Machine            aMachine)
{
  auto tLowerPDE = Plato::get_pde_type(aProbParams);
#ifdef PLATO_ELLIPTIC
  if (tLowerPDE == "elliptic"){
    return ( makeProblem<Plato::Elliptic::Problem,Plato::Elliptic::Nonlinear::Mechanics>(aMesh, aProbParams, aMachine) );
  }
#endif
  else{
    return nullptr;
  }
}

/// @fn create_linear_electrical_problem
/// @brief inline function, create electrical residual evaluator
/// @param [in] aMesh      contains mesh and model information
/// @param [in] aProbParams inpur problem parameters
/// @param [in] aMachine   contains mpi communicator information
/// @return shared pointer to residual evaluator
inline
std::shared_ptr<Plato::AbstractProblem>
create_linear_electrical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aProbParams,
 Comm::Machine            aMachine)
{
  auto tLowerPDE = Plato::get_pde_type(aProbParams);
#ifdef PLATO_ELLIPTIC
  if (tLowerPDE == "elliptic"){
    return ( makeProblem<Plato::Elliptic::Problem,Plato::Elliptic::Linear::Electrical>(aMesh, aProbParams, aMachine) );
  }
#endif
  else{
    return nullptr;
  }
}

/******************************************************************************//**
* \brief Create plasticity problem.
* \param [in] aMesh        Plato mesh database
* \param [in] aProbParams input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type plasticity
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_plasticity_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aProbParams,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::get_pde_type(aProbParams);

#ifdef PLATO_ELLIPTIC
#ifdef PLATO_PLASTICITY
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < PlasticityProblem<::Plato::InfinitesimalStrainPlasticity<SpatialDim>> > (aMesh, aProbParams, aMachine);
        tOutput->readEssentialBoundaryConditions(aProbParams);
        return tOutput;
    }
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
* \param [in] aProbParams input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type thermoplasticity
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_thermoplasticity_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aProbParams,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::get_pde_type(aProbParams);

#ifdef PLATO_ELLIPTIC
#ifdef PLATO_PLASTICITY
    if(tLowerPDE == "elliptic")
    {
        auto tOutput = std::make_shared < PlasticityProblem<::Plato::InfinitesimalStrainThermoPlasticity<SpatialDim>> > (aMesh, aProbParams, aMachine);
        tOutput->readEssentialBoundaryConditions(aProbParams);
        return tOutput;
    }
#endif
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_thermoplasticity_problem

/******************************************************************************//**
* \brief Create a abstract problem of type thermal.
* \param [in] aMesh      mesh metadata
* \param [in] aProbParams input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type thermal
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_linear_thermal_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aProbParams,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::get_pde_type(aProbParams);

#ifdef PLATO_PARABOLIC
    if(tLowerPDE == "parabolic")
    {
        return makeProblem<Plato::Parabolic::Problem, Plato::Parabolic::Linear::Thermal>(aMesh, aProbParams, aMachine);
    }
#endif
#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Elliptic::Linear::Thermal>(aMesh, aProbParams, aMachine);
    }
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_linear_thermal_problem

/******************************************************************************//**
* \brief Create a abstract problem of type electromechanical.
* \param [in] aMesh      mesh metadata
* \param [in] aProbParams input xml metadata
* \param [in] aMachine   mpi communicator interface
* \returns shared pointer to abstract problem of type electromechanical
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_linear_electromechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aProbParams,
 Comm::Machine            aMachine)
 {
     auto tLowerPDE = Plato::get_pde_type(aProbParams);

#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Elliptic::Linear::Electromechanics>(aMesh, aProbParams, aMachine);
    }
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_linear_electromechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type thermomechanical.
* \param [in] aMesh        mesh metadata
* \param [in] aProbParams input xml metadata
* \param [in] aMachine     mpi communicator interface
* \returns shared pointer to abstract problem of type thermomechanical
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_linear_thermomechanical_problem
(Plato::Mesh              aMesh,
 Teuchos::ParameterList & aProbParams,
 Comm::Machine            aMachine)
 {
    auto tLowerPDE = Plato::get_pde_type(aProbParams);

#ifdef PLATO_PARABOLIC
    if(tLowerPDE == "parabolic")
    {
      return ( 
        makeProblem<Plato::Parabolic::Problem, Plato::Parabolic::Linear::Thermomechanics>(aMesh,aProbParams,aMachine) 
      );
    }
#endif
#ifdef PLATO_ELLIPTIC
    if(tLowerPDE == "elliptic")
    {
      return ( 
        makeProblem<Plato::Elliptic::Problem, Plato::Elliptic::Linear::Thermomechanics>(aMesh,aProbParams,aMachine) 
      );
    }
#endif
    {
      ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_linear_thermomechanical_problem

/******************************************************************************//**
* \brief Create a abstract problem of type incompressible fluid.
* \param [in] aMesh      mesh metadata
* \param [in] aProbParams input xml metadata
* \param [in] aMachine   mpi wrapper 
* \returns shared pointer to abstract problem of type incompressible fluid
**********************************************************************************/
inline
std::shared_ptr<Plato::AbstractProblem>
create_incompressible_fluid_problem(
  Plato::Mesh            & aMesh,
  Teuchos::ParameterList & aProbParams,
  Comm::Machine          & aMachine)
 {
    auto tLowerPDE = Plato::get_pde_type(aProbParams);

#ifdef PLATO_HYPERBOLIC
#ifdef PLATO_FLUIDS
    if (tLowerPDE == "hyperbolic")
    {
        return makeProblem<Plato::Fluids::QuasiImplicit, Plato::IncompressibleFluids>(aMesh, aProbParams, aMachine);
    }
#endif
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
 }
 // function create_incompressible_fluid_problem

/// @class ProblemFactory
/// @brief create the problem evaluator facilitating analysis and optimization
class ProblemFactory
{
public:
  /// @brief create problem evaluator 
  /// @param [in] aMesh        mesh database
  /// @param [in] aInputParams input problem parameters
  /// @param [in] aMachine     mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  create(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aInputParams,
    Comm::Machine          & aMachine
  )
  {
    if( !aInputParams.isSublist("Plato Problem") ){
      ANALYZE_THROWERR("ERROR: Parameter list ('Plato Problem') is not defined, analysis cannot be performed")
    }
    auto aProbParams = aInputParams.sublist("Plato Problem");

    if( !aProbParams.isParameter("Physics") ){
      ANALYZE_THROWERR(std::string("ERROR: Argument ('Physics') is not defined, ") 
        + "physics equations cannot be deduced")
    }
    auto tPhysics = aProbParams.get< std::string >("Physics");

    Plato::PhysicsEnum tS2E;
    if( tS2E.production(tPhysics) ){
      return ( this->createSupportedProblem(aMesh,aProbParams,aMachine) );
    }
    else 
    if( !tS2E.production(tPhysics) ){
      return ( this->createExperimentalProblem(aMesh,aProbParams,aMachine) );
    }
    else{
      ANALYZE_THROWERR("ERROR: Did not create problem evaluator!! Check input deck setup");
      return nullptr;
    }
  }

private:
  /// createSupportedProblem
  /// @brief create supported problem evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createSupportedProblem(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine          & aMachine
  )
  {
    Plato::PhysicsEnum tS2E;
    auto tResponse = aProbParams.get<std::string>("Response","linear");
    auto tSupportedResponse = tS2E.response(tResponse);
    switch (tSupportedResponse)
    {
      case Plato::response_t::NONLINEAR:
        return ( this->createSupportedNonlinearPhysics(aMesh,aProbParams,aMachine) );
        break;
      default:
      case Plato::response_t::LINEAR:
        return ( this->createSupportedLinearPhysics(aMesh,aProbParams,aMachine) );
        break;
    }
  }

  /// createSupportedLinearPhysics
  /// @brief create supported linear physics evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createSupportedLinearPhysics(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine          & aMachine
  )
  {
    Plato::PhysicsEnum tS2E;
    auto tPhysics = aProbParams.get< std::string >("Physics");
    auto tSupportedPhysics = tS2E.physics(tPhysics);
    switch (tSupportedPhysics)
    {
    case Plato::physics_t::MECHANICAL:
      return ( Plato::create_linear_mechanical_problem(aMesh,aProbParams,aMachine) );
      break;
    case Plato::physics_t::THERMAL:
      return ( Plato::create_linear_thermal_problem(aMesh,aProbParams,aMachine) );
      break;
    case Plato::physics_t::ELECTRICAL: 
      return ( Plato::create_linear_electrical_problem(aMesh,aProbParams,aMachine) );
      break;
    case Plato::physics_t::THERMOMECHANICAL:
      return ( Plato::create_linear_thermomechanical_problem(aMesh,aProbParams,aMachine) );
      break;
    default:
      return nullptr;
      break;
    }
  }

  /// createSupportedNonlinearPhysics
  /// @brief create supported nonlinear physics evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createSupportedNonlinearPhysics(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine          & aMachine
  )
  {
    Plato::PhysicsEnum tS2E;
    auto tPhysics = aProbParams.get< std::string >("Physics");
    auto tSupportedPhysics = tS2E.physics(tPhysics);
    switch (tSupportedPhysics)
    {
    case Plato::physics_t::MECHANICAL:
      return ( this->createNonlinearMechanicalProblem(aMesh,aProbParams,aMachine) );
      break;
    case Plato::physics_t::THERMOMECHANICAL:
      return ( this->createNonlinearThermoMechanicalProblem(aMesh,aProbParams,aMachine) );
      break;
    default:
      return nullptr;
      break;
    }
  }

  /// createNonlinearMechanicalProblem
  /// @brief create nonlinear mechanical physics evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createNonlinearMechanicalProblem(
    Plato::Mesh              aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine            aMachine
  )
  {
    auto tLowerPDE = Plato::get_pde_type(aProbParams);
  #ifdef PLATO_ELLIPTIC
    if (tLowerPDE == "elliptic")
    {
      return makeProblem<Plato::Elliptic::Problem,Plato::Elliptic::Nonlinear::Mechanics>(aMesh,aProbParams,aMachine);
    }
  #endif
    {
      ANALYZE_THROWERR(std::string("Argument ('PDE Constraint = ") + tLowerPDE + "') is not supported.");
    }
  }

  /// createNonlinearThermoMechanicalProblem
  /// @brief create nonlinear thermomechanical physics evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createNonlinearThermoMechanicalProblem(
    Plato::Mesh              aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine            aMachine
  )
  {
    auto tLowerPDE = Plato::get_pde_type(aProbParams);
  #ifdef PLATO_ELLIPTIC
    if (tLowerPDE == "elliptic")
    {
      return makeProblem<Plato::Elliptic::Problem,Plato::Elliptic::Nonlinear::ThermoMechanics>(
        aMesh,aProbParams,aMachine
      );
    }
  #endif
    {
      ANALYZE_THROWERR(std::string("Argument ('PDE Constraint = ") + tLowerPDE + "') is not supported.");
    }
  }

  /// createExperimentalProblem
  /// @brief create experimental physics evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createExperimentalProblem(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine          & aMachine
  )
  {
    Plato::PhysicsEnum tS2E;
    auto tResponse = aProbParams.get<std::string>("Response","linear");
    auto tSupportedResponse = tS2E.response(tResponse);
    auto tPhysics = aProbParams.get< std::string >("Physics");
    auto tSupportedPhysics = tS2E.physics(tPhysics);

    tSupportedResponse = (tSupportedPhysics == Plato::physics_t::PLASTICITY) || 
                         (tSupportedPhysics == Plato::physics_t::THERMOPLASTICITY) ? 
                         Plato::response_t::NONLINEAR : tSupportedResponse;

    switch (tSupportedResponse)
    {
      case Plato::response_t::NONLINEAR:
        return ( this->createExperimentalNonlinearPhysics(aMesh,aProbParams,aMachine) );
        break;
      default:
      case Plato::response_t::LINEAR:
        return ( this->createExperimentalLinearPhysics(aMesh,aProbParams,aMachine) );
        break;
    }
  }

  /// createExperimentalLinearPhysics
  /// @brief create experimental linear physics evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createExperimentalLinearPhysics(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine          & aMachine
  )
  {
    Plato::PhysicsEnum tS2E;
    auto tPhysics = aProbParams.get< std::string >("Physics");
    auto tSupportedPhysics = tS2E.physics(tPhysics);
    switch (tSupportedPhysics)
    {
    case Plato::physics_t::HELMHOLTZ_FILTER:
      return ( this->createHelmholtzFilter(aMesh,aProbParams,aMachine) );
      break;
    case Plato::physics_t::ELECTROMECHANICAL:
      return ( Plato::create_linear_electromechanical_problem(aMesh,aProbParams,aMachine) );
      break;
    case Plato::physics_t::INCOMPRESSIBLE_FLUID: 
      return ( Plato::create_incompressible_fluid_problem(aMesh,aProbParams,aMachine) );
      break;
    default:
      return nullptr;
      break;
    }
  }

  /// createExperimentalNonlinearPhysics
  /// @brief create experimental nonlinear physics evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createExperimentalNonlinearPhysics(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine          & aMachine
  )
  {
    Plato::PhysicsEnum tS2E;
    auto tPhysics = aProbParams.get< std::string >("Physics");
    auto tSupportedPhysics = tS2E.physics(tPhysics);
    switch (tSupportedPhysics)
    {
    case Plato::physics_t::PLASTICITY:
      return ( Plato::create_plasticity_problem(aMesh,aProbParams,aMachine) );
      break;
    case Plato::physics_t::THERMOPLASTICITY:
      return ( Plato::create_thermoplasticity_problem(aMesh,aProbParams,aMachine) );
      break;
    default:
      return nullptr;
      break;
    }
  }

  /// createHelmholtzFilter
  /// @brief create helmholtz problem evaluator
  /// @param [in] aMesh       mesh database
  /// @param [in] aProbParams input problem parameters
  /// @param [in] aMachine    mpi wrapper
  /// @return standard shared pointer to problem evaluator
  std::shared_ptr<Plato::AbstractProblem>
  createHelmholtzFilter(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aProbParams,
    Comm::Machine          & aMachine
  )
  {
#ifdef PLATO_HELMHOLTZ
    return ( Plato::makeProblem<Plato::Helmholtz::Problem,Plato::HelmholtzFilter>(aMesh,aProbParams,aMachine) );
#endif
  }
};
// class ProblemFactory

}
// namespace Plato
