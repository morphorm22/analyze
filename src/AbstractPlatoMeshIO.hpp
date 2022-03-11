#pragma once

#include "PlatoStaticsTypes.hpp"

#include <vector>
#include <string>

namespace Plato
{
    class AbstractMeshIO
    {
        public:
            /******************************************************************************//**
            * \brief Return number of nodes in the mesh
            * If the mesh is distributed, return the number of local nodes
            **********************************************************************************/
            virtual Plato::OrdinalType NumNodes() const = 0;

            /******************************************************************************//**
            * \brief Return number of elements in the mesh
            * If the mesh is distributed, return the number of local elements
            **********************************************************************************/
            virtual Plato::OrdinalType NumElements() const = 0;

            /******************************************************************************//**
            * \brief Add node data to be written
            * \param [in] aName Name of the node data
            * \param [in] aData Data to be written.  Given in form {d01, d02, ..., d0n, d1n, ..., dNn}
            *             where N is the number of nodes and n is aNumDofs.
            * \param [in] aNumDofs Number of degrees of freedom per node
            **********************************************************************************/
            virtual void AddNodeData(std::string aName, Plato::ScalarVector aData, Plato::OrdinalType aNumDofs) = 0;


            /******************************************************************************//**
            * \brief Add node data to be written
            * \param [in] aName Name of the node data
            * \param [in] aData Data to be written.  Given in form {d01, d02, ..., d0n, d1n, ..., dNn}
            *             where N is the number of nodes and n is aNumDofs.
            * \param [in] aDofNames Names of degrees of freedom
            **********************************************************************************/
            virtual void AddNodeData(std::string aName, Plato::ScalarVector aData, std::vector<std::string> aDofNames) = 0;

            /******************************************************************************//**
            * \brief Add node data to be written
            * \param [in] aName Name of the node data
            * \param [in] aData Data to be written.  
            **********************************************************************************/
            virtual void AddNodeData(std::string aName, Plato::ScalarVector aData) = 0;

            /******************************************************************************//**
            * \brief Add element data to be written
            * \param [in] aName Name of the element data
            * \param [in] aData Data to be written.  Given in form aData(N) where N is the 
            *                   number of elements.
            **********************************************************************************/
            virtual void AddElementData(std::string aName, Plato::ScalarVector aData) = 0;

            /******************************************************************************//**
            * \brief Add element data to be written
            * \param [in] aName Name of the element data
            * \param [in] aData Data to be written.  Given in form aData(n,N) where N is the 
            *                   number of elements, and n is the number of terms.
            * Note: Names are inferred from the number of terms:
            * n=1: scalar. Name is aName.
            * n=nSpaceDims: vector. Names are aName+' X', aName+' Y', etc.
            * n=nVoigtDims: symtensor. Names are aName+' '+{'XX','YY','ZZ','YZ','XZ','XY'} for 3D.
            **********************************************************************************/
            virtual void AddElementData(std::string aName, Plato::ScalarMultiVector aData) = 0;

            /******************************************************************************//**
            * \brief Write data to mesh file
            * \param [in] aStepIndex Step index to be written.
            * \param [in] aTimeValue Current time.
            * \param [in] aMode File IO mode, i.e., "Read", "Write", "Append".
            **********************************************************************************/
            virtual void Write(
                Plato::OrdinalType aStepIndex,
                Plato::Scalar      aTimeValue) = 0;

            /******************************************************************************//**
            * \brief Get the number of time steps
            **********************************************************************************/
            virtual Plato::OrdinalType NumTimeSteps() = 0;

            /******************************************************************************//**
            * \brief Read node scalar field
            * \param [in] aVariableName Name of the node field to read.
            * \param [in] aStepIndex Step index to be read.
            **********************************************************************************/
            virtual Plato::ScalarVector ReadNodeData(
                const std::string        & aVariableName,
                      Plato::OrdinalType   aStepIndex) = 0;
    };
}
