#pragma once

#include <memory>

#include "OmegaHMesh.hpp"
#include "OmegaHMeshIO.hpp"

#include "EngineMesh.hpp"
#include "EngineMeshIO.hpp"

namespace Plato {

#ifdef USE_OMEGAH_MESH
    using MeshType = OmegaHMesh;
    using MeshIOType = OmegaHMeshIO;
#else
    using MeshType = EngineMesh;
    using MeshIOType = EngineMeshIO;
#endif

    using Mesh = std::shared_ptr<Plato::MeshType>;
    namespace MeshFactory
    {
        inline void initialize(int& aArgc, char**& aArgv)
        {
            Plato::OmegaH::Library = new Omega_h::Library(&aArgc, &aArgv);
        }
        inline Plato::Mesh create(std::string aFilePath)
        {
            return std::make_shared<Plato::MeshType>(aFilePath);
        }
        inline void finalize()
        {
            if(Plato::OmegaH::Library) delete Plato::OmegaH::Library;
        }
    }
    // end namespace MeshFactory

    using MeshIO = std::shared_ptr<Plato::MeshIOType>;
    namespace MeshIOFactory
    {
        inline Plato::MeshIO create(std::string aFilePath, Plato::Mesh aMesh, std::string aMode)
        {
            return std::make_shared<Plato::MeshIOType>(aFilePath, *aMesh, aMode);
        }
    }
    // end namespace MeshIOFactory

} // end namespace Plato
