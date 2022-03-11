#ifndef PLATO_DRIVER_HPP
#define PLATO_DRIVER_HPP

#include <string>
#include <vector>
#include <memory>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoMesh.hpp"
#include "OmegaHMesh.hpp"
#include "AnalyzeOutput.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoProblemFactory.hpp"
//#include "StructuralDynamicsOutput.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Run simulation with Plato Analyze.
 *
 * \tparam SpatialDim spatial dimensions
 *
 * \param [in] aInputData   input parameters list
 * \param [in] aMesh        mesh database
*******************************************************************************/
template<const Plato::OrdinalType SpatialDim>
void run(
    Teuchos::ParameterList& aInputData,
    Comm::Machine           aMachine,
    Plato::Mesh             aMesh)
{
    // create mesh based density from host data
    std::vector<Plato::Scalar> tControlHost(aMesh->NumNodes(), 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tControlHostView(tControlHost.data(), tControlHost.size());
    auto tControl = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tControlHostView);

    // Solve Plato problem
    Plato::ProblemFactory<SpatialDim> tProblemFactory;
    std::shared_ptr<::Plato::AbstractProblem> tPlatoProblem = tProblemFactory.create(aMesh, aInputData, aMachine);
    auto tSolution = tPlatoProblem->solution(tControl);
    if(false){ tSolution.print(); }

    auto tPlatoProblemList = aInputData.sublist("Plato Problem");
    if (tPlatoProblemList.isSublist("Criteria"))
    {
        auto tCriteriaList = tPlatoProblemList.sublist("Criteria");
        for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaList.begin(); tIndex != tCriteriaList.end(); ++tIndex)
        {
            std::string tName = tCriteriaList.name(tIndex);
            Plato::Scalar tCriterionValue = tPlatoProblem->criterionValue(tControl, tSolution, tName);
            printf("Criterion '%s' , Value %0.10e\n", tName.c_str(), tCriterionValue);
        }
    }

    auto tFilepath = aInputData.get<std::string>("Output Viz");
    tPlatoProblem->output(tFilepath);
}
// function run


/***************************************************************************//**
 * \brief Run simulation with Plato Analyze.
 *
 * \tparam SpatialDim spatial dimensions
 *
 * \param [in] aInputData   input parameters list
 * \param [in] aInputFile   Plato Analyze input file name
*******************************************************************************/
template<const Plato::OrdinalType SpatialDim>
void driver(
    Teuchos::ParameterList & aInputData,
    Comm::Machine            aMachine)
{
    auto tInputMesh = aInputData.get<std::string>("Input Mesh");

    Plato::Mesh tMesh = Plato::MeshFactory::create(tInputMesh);

    Plato::run<SpatialDim>(aInputData, aMachine, tMesh);
}
// function driver

/***************************************************************************//**
 * \brief Run simulation with Plato Analyze.
 *
 * \param [in] aInputData   input parameters list
 * \param [in] aInputFile   Plato Analyze input file name
*******************************************************************************/
void driver(
    Teuchos::ParameterList & aInputData,
    Comm::Machine            aMachine)
{
    const Plato::OrdinalType tSpaceDim = aInputData.get<Plato::OrdinalType>("Spatial Dimension", 3);

    // Run Plato problem
    if(tSpaceDim == static_cast<Plato::OrdinalType>(3))
    {
        #ifdef PLATOANALYZE_3D
        driver<3>(aInputData, aMachine);
        #else
        throw std::runtime_error("3D physics option is not compiled.");
        #endif
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(2))
    {
        #ifdef PLATOANALYZE_2D
        driver<2>(aInputData, aMachine);
        #else
        throw std::runtime_error("2D physics option is not compiled.");
        #endif
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(1))
    {
        #ifdef PLATOANALYZE_1D
        driver<1>(aInputData, aMachine);
        #else
        throw std::runtime_error("1D physics option is not compiled.");
        #endif
    }
}
// function driver


}
// namespace Plato

#endif /* #ifndef PLATO_DRIVER_HPP */

