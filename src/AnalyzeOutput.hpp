#ifndef PLATO_OUTPUT_HPP
#define PLATO_OUTPUT_HPP

#include <string>

#include <Teuchos_ParameterList.hpp>

#include "Solutions.hpp"
#include "PlatoUtilities.hpp"

namespace Plato
{
    inline void
    AddStateData(
        Plato::MeshIO      aWriter,
        Plato::DataMap     aDataMap,
        Plato::OrdinalType aNumDims
    )
    {
        auto tNumNodes = aWriter->NumNodes();
        auto tNumElements = aWriter->NumElements();

        {   // ScalarVectors
            //
            for( auto& tPair : aDataMap.scalarVectors )
            {
                if(tPair.second.extent(0) == tNumElements)
                {   
                    aWriter->AddElementData(tPair.first, tPair.second);
                }
            }
        }
        {   // ScalarMultiVectors
            //
            for( auto& tPair : aDataMap.scalarMultiVectors )
            {
                if(tPair.second.extent(0) == tNumElements)
                {   
                    aWriter->AddElementData(tPair.first, tPair.second);
                }
            }
        }
        {   // Node Scalar
            //
            for( auto& tPair : aDataMap.scalarNodeFields )
            {
                if(tPair.second.extent(0) == tNumNodes)
                {   
                    aWriter->AddNodeData(tPair.first, tPair.second);
                }
            }
        }
        {   // Node Vector
            //
            for( auto& tPair : aDataMap.vectorNodeFields )
            {
                if(tPair.second.extent(0) == aNumDims*tNumNodes)
                {   
                    aWriter->AddNodeData(tPair.first, tPair.second, aNumDims);
                }
            }
        }
    }

    /******************************************************************************/ /**
    * \brief Output data for all your output needs
    * \param [in] aOutputFilePath  output viz file path
    * \param [in] aSolutionsOutput global solution data for output
    * \param [in] aStateDataMap    Plato Analyze data map
    * \param [in] aMesh            mesh database
    **********************************************************************************/
    inline void
    universal_solution_output(
        const std::string      & aOutputFilePath,
        const Plato::Solutions & aSolutionsOutput,
        const Plato::DataMap   & aStateDataMap,
              Plato::Mesh        aMesh)
    {
        auto tWriter = Plato::MeshIOFactory::create(aOutputFilePath, aMesh, "Write");

        auto tNumTimeSteps = aSolutionsOutput.getNumTimeSteps();
        auto tNumStates = aStateDataMap.stateDataMaps.size();

        bool tWriteStates = (tNumStates == tNumTimeSteps);
        if (!tWriteStates)
        {
            REPORT("State data not provided by physics so not written to output file.");
        }

        for (Plato::OrdinalType tStepIndex = 0; tStepIndex < tNumTimeSteps; ++tStepIndex)
        {
            for (auto &tSolutionOutputName : aSolutionsOutput.tags())
            {
                auto tSolutions = aSolutionsOutput.get(tSolutionOutputName);
                auto tDofNames = aSolutionsOutput.getDofNames(tSolutionOutputName);
                Plato::ScalarVector tSolution = Kokkos::subview(tSolutions, tStepIndex, Kokkos::ALL());

                if (tDofNames.size() == 0) // dof names not provided
                {
                    auto tNumDofs = aSolutionsOutput.getNumDofs(tSolutionOutputName);
                    tWriter->AddNodeData(tSolutionOutputName, tSolution, tNumDofs);
                }
                else
                {
                    tWriter->AddNodeData(tSolutionOutputName, tSolution, tDofNames);
                }
            }

            if (tWriteStates)
            {
                AddStateData(tWriter, aStateDataMap.getState(tStepIndex), aMesh->NumDimensions());
            }

            tWriter->Write(/*time_index*/ tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex);
        }
    }
}
// namespace Plato

#endif
