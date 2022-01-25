#pragma once

#include <memory>

#include "PlatoMesh.hpp"
#include "AbstractPlatoMeshIO.hpp"

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>

namespace Plato
{
    class OmegaHMeshIO : public AbstractMeshIO
    {
        Omega_h::Mesh *mMesh;

        std::shared_ptr<Omega_h::vtk::Writer> mWriter;

        std::vector<Omega_h::filesystem::path> mPaths;


        enum struct Mode { Read, Write, Append };

        Mode mMode;

        public:
            OmegaHMeshIO(std::string aOutputFilePath, Plato::OmegaHMesh & aMesh, std::string aMode="Write");

            Plato::OrdinalType NumNodes() const override;
            Plato::OrdinalType NumElements() const override;

            void AddNodeData(std::string aName, Plato::ScalarVector aData) override;
            void AddNodeData(std::string aName, Plato::ScalarVector aData, Plato::OrdinalType aNumdofs) override;
            void AddNodeData(std::string aName, Plato::ScalarVector aData, std::vector<std::string> aDofNames) override;

            void AddElementData(std::string aName, Plato::ScalarVector aData) override;
            void AddElementData(std::string aName, Plato::ScalarMultiVector aData) override;

            void Write(Plato::OrdinalType aStepIndex, Plato::Scalar aTimeValue) override;

            Plato::OrdinalType NumTimeSteps() override;

            Plato::ScalarVector ReadNodeData( const std::string & aVariableName, Plato::OrdinalType aStepIndex) override;
    };
}
