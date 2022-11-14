#ifndef PLATO_DRIVER_HPP
#define PLATO_DRIVER_HPP

#include <string>
#include <vector>
#include <memory>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoMesh.hpp"
#include "AnalyzeOutput.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoProblemFactory.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Run simulation with Plato Analyze.
 *
 * \tparam SpatialDim spatial dimensions
 *
 * \param [in] aInputData   input parameters list
 * \param [in] aInputFile   Plato Analyze input file name
*******************************************************************************/
void driver(
    Teuchos::ParameterList & aInputData,
    Comm::Machine            aMachine)
{
    auto tInputMesh = aInputData.get<std::string>("Input Mesh");

    Plato::Mesh tMesh = Plato::MeshFactory::create(tInputMesh);

    // create default control vector
    Plato::ScalarVector tControl("control", tMesh->NumNodes());
    Kokkos::deep_copy(tControl, 1.0);

    // Solve Plato problem
    Plato::ProblemFactory tProblemFactory;
    std::shared_ptr<::Plato::AbstractProblem> tPlatoProblem = tProblemFactory.create(tMesh, aInputData, aMachine);
    auto tSolution = tPlatoProblem->solution(tControl);

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
// function driver

}
// namespace Plato

#endif /* #ifndef PLATO_DRIVER_HPP */

