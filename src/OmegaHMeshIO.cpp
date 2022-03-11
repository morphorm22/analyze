#include "UtilsOmegaH.hpp"
#include "AnalyzeAppUtils.hpp"
#include "OmegaHMeshIO.hpp"
#include "OmegaHMesh.hpp"

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>

namespace Plato
{
    OmegaHMeshIO::OmegaHMeshIO(
        std::string         aFilePath,
        Plato::OmegaHMesh & aMesh,
        std::string         aMode
    )
    {
        auto tModeLower = Plato::tolower(aMode);
        if( tModeLower == "write" || tModeLower == "w")
        {
            mMode = Mode::Write;
            mMesh = &aMesh.mMesh;
            mWriter = std::make_shared<Omega_h::vtk::Writer>(aFilePath, mMesh, mMesh->dim(), 0.0);
        }
        else
        if( tModeLower == "read" || tModeLower == "r")
        {
            mMode = Mode::Read;
            mPaths = Plato::omega_h::read_pvtu_file_paths(aFilePath);
            mMesh = nullptr;
        }
        else
        if( tModeLower == "append" || tModeLower == "a")
        {
            ANALYZE_THROWERR("OmegaHMesh: 'Append' IO mode not implemented");
        }
    }

    Plato::OrdinalType OmegaHMeshIO::NumNodes()    const {return mMesh->nverts();}
    Plato::OrdinalType OmegaHMeshIO::NumElements() const {return mMesh->nelems();}

    void OmegaHMeshIO::AddNodeData(
        std::string              aName,
        Plato::ScalarVector      aData
    )
    {
        AddNodeData(aName, aData, 1);
    }
    void OmegaHMeshIO::AddNodeData(
        std::string              aName,
        Plato::ScalarVector      aData,
        std::vector<std::string> aDofNames
    )
    {
        auto tStride = aDofNames.size();
        for( decltype(tStride) tIndex=0; tIndex<tStride; tIndex++)
        {
            auto tScalarVector = Plato::get_vector_component(aData, tIndex, tStride);
            AddNodeData(aDofNames[tIndex], tScalarVector);
        }

    }
    void OmegaHMeshIO::AddNodeData(
        std::string         aName,
        Plato::ScalarVector aData,
        Plato::OrdinalType  aNumDofs
    )
    {
        Omega_h::Write<Omega_h::Real> tDataOmegaH(aData.size(), "OmegaH Copy");
        Plato::copy_1Dview_to_write(aData, tDataOmegaH);
        mMesh->add_tag(Omega_h::VERT, aName, aNumDofs, Omega_h::Reals(tDataOmegaH));
    }

    void OmegaHMeshIO::AddElementData(
        std::string         aName,
        Plato::ScalarVector aData
    )
    {
        Omega_h::Write<Omega_h::Real> tDataWrite(aData.size(), aName);
        Plato::copy_1Dview_to_write(aData, tDataWrite);
        mMesh->add_tag(mMesh->dim(), aName, /*numDataPerElement=*/1, Omega_h::Reals(tDataWrite));
    }

    void OmegaHMeshIO::AddElementData(
        std::string         aName,
        Plato::ScalarMultiVector aData
    )
    {
        auto tNumElements = aData.extent(0);
        auto tNumDataPerElement = aData.extent(1);
        auto tNumData = tNumElements * tNumDataPerElement;
        Omega_h::Write<Omega_h::Real> tDataWrite(tNumData, aName);
        Plato::copy_2Dview_to_write(aData, tDataWrite);
        mMesh->add_tag(mMesh->dim(), aName, tNumDataPerElement, Omega_h::Reals(tDataWrite));
    }
    
    void OmegaHMeshIO::Write(
        Plato::OrdinalType aStepIndex,
        Plato::Scalar      aTimeValue
    )
    {
        Omega_h::TagSet tTagsOmegaH = Omega_h::vtk::get_all_vtk_tags(mMesh, mMesh->dim());
        mWriter->write(aStepIndex, aTimeValue, tTagsOmegaH);
    }

    Plato::OrdinalType
    OmegaHMeshIO::NumTimeSteps()
    {
        return mPaths.size();
    }


    Plato::ScalarVector
    OmegaHMeshIO::ReadNodeData(
        const std::string        & aVariableName,
              Plato::OrdinalType   aStepIndex
    )
    {
        auto tPath = mPaths[aStepIndex];
        Omega_h::Mesh tMesh(OmegaH::Library);
        Omega_h::vtk::read_parallel(tPath, OmegaH::Library->world(), &tMesh);
        return Plato::omega_h::read_metadata_from_mesh(tMesh, /*EntityDim=*/Omega_h::VERT, aVariableName);
    }
}
