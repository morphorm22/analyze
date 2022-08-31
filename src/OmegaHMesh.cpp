#include "OmegaHMesh.hpp"
#include "UtilsOmegaH.hpp"

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>

namespace Plato
{
    namespace OmegaH
    {
        Omega_h::Library* Library = nullptr;
    }
    OmegaHMesh::OmegaHMesh(
        std::string aInputMeshName
    ) :
        mMesh(Omega_h::read_mesh_file(aInputMeshName, OmegaH::Library->world())),
        mFileName(aInputMeshName)
    {
        mMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

        mAssoc[Omega_h::ELEM_SET] = mMesh.class_sets;
        mAssoc[Omega_h::NODE_SET] = mMesh.class_sets;
        mAssoc[Omega_h::SIDE_SET] = mMesh.class_sets;
        mMeshSets = Omega_h::invert(&mMesh, mAssoc);

        initialize();
    }

    void
    OmegaHMesh::initialize()
    {
       if (mMesh.dim() == 3)
       {
           this->InvertSideSets<3>();
           mElementType = "TET4";
       }
       else
       if (mMesh.dim() == 2)
       {
           this->InvertSideSets<2>();
           mElementType = "TRI3";
       }
       else
       if (mMesh.dim() == 1)
       {
           this->InvertSideSets<1>();
           mElementType = "BAR2";
       }
       else
       {
           throw std::runtime_error("Unsupported mesh dimension");
       }
    }

    Plato::ScalarVectorT<const Plato::OrdinalType>
    OmegaHMesh::GetLocalElementIDs(
        std::string aBlockName
    ) const
    {
        auto& tElemSets = mMeshSets[Omega_h::ELEM_SET];
        auto tElemSetsI = tElemSets.find(aBlockName);
        if(tElemSetsI == tElemSets.end())
        {
            std::ostringstream tMsg;
            tMsg << "Did not find element set (i.e. element block) with name = '" << aBlockName << "'.";
            ANALYZE_THROWERR(tMsg.str())
        }

        return tElemSetsI->second.view();
    }

    std::vector<std::string>
    OmegaHMesh::GetElementBlockNames() const 
    {
        std::vector<std::string> tRetVal;
        for(const auto& tPair : mMeshSets[Omega_h::ELEM_SET])
        {
            tRetVal.push_back(tPair.first);
        }
        return tRetVal;
    }

    std::vector<std::string>
    OmegaHMesh::GetNodeSetNames() const 
    {
        std::vector<std::string> tRetVal;
        for(const auto& tPair : mMeshSets[Omega_h::NODE_SET])
        {
            tRetVal.push_back(tPair.first);
        }
        return tRetVal;
    }

    std::vector<std::string>
    OmegaHMesh::GetSideSetNames() const 
    {
        std::vector<std::string> tRetVal;
        for(const auto& tPair : mMeshSets[Omega_h::SIDE_SET])
        {
            tRetVal.push_back(tPair.first);
        }
        return tRetVal;
    }

    std::string
    OmegaHMesh::FileName() const { return mFileName; }

    std::string
    OmegaHMesh::ElementType() const { return mElementType; }

    Plato::OrdinalType
    OmegaHMesh::NumNodes() const { return mMesh.nverts(); }

    Plato::OrdinalType
    OmegaHMesh::NumElements() const { return mMesh.nelems(); }

    Plato::OrdinalType
    OmegaHMesh::NumNodesPerElement() const { return mMesh.dim()+1; }

    Plato::OrdinalType
    OmegaHMesh::NumDimensions() const { return mMesh.dim(); }

    Plato::ScalarVectorT<const Plato::Scalar>
    OmegaHMesh::Coordinates() const
    {
        return mMesh.coords().view();
    }

    void
    OmegaHMesh::SetCoordinates(
        Plato::ScalarVector aCoordinates
    )
    {
        mMesh.set_coords(Omega_h::Read<Plato::Scalar>(Omega_h::Write<Plato::Scalar>(aCoordinates)));
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    OmegaHMesh::Connectivity()
    {
        return mMesh.ask_elem_verts().view();
    }

    void
    OmegaHMesh::NodeNodeGraph(
        Plato::OrdinalVectorT<const Plato::OrdinalType> & aOffsetMap,
        Plato::OrdinalVectorT<const Plato::OrdinalType> & aNodeOrds
    )
    {
        const Plato::OrdinalType cVertexDim = 0;
        auto tNode2nodes = mMesh.ask_star(cVertexDim);
        aOffsetMap = tNode2nodes.a2ab.view();
        aNodeOrds = tNode2nodes.ab2b.view();

        return;
    }

    void
    OmegaHMesh::NodeElementGraph(
        Plato::OrdinalVectorT<const Plato::OrdinalType> & aOffsetMap,
        Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds
    )
    {
        const Plato::OrdinalType cVertexDim = 0;
        auto tNode2elems = mMesh.ask_up(cVertexDim, mMesh.dim());
        aOffsetMap = tNode2elems.a2ab.view();
        aElementOrds = tNode2elems.ab2b.view();

        return;
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    OmegaHMesh::GetSideSetFaces( std::string aSideSetName) const
    {
        return mSideSetFacesOrdinals.at(aSideSetName);
    }
            
    Plato::OrdinalVectorT<const Plato::OrdinalType>
    OmegaHMesh::GetSideSetElements( std::string aSideSetName) const
    {
        return mSideSetElementsOrdinals.at(aSideSetName);
    }
            
    Plato::OrdinalVectorT<const Plato::OrdinalType>
    OmegaHMesh::GetSideSetLocalNodes( std::string aSideSetName) const
    {
        return mSideSetLocalNodesOrdinals.at(aSideSetName);
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    OmegaHMesh::GetSideSetElementsComplement( std::vector<std::string> aExcludeNames)
    {
        if( mSideSetElementsComplementOrdinals.count(aExcludeNames) )
        {
            return mSideSetElementsComplementOrdinals.at(aExcludeNames);
        }
        else
        {
            createComplement( aExcludeNames );
            return mSideSetElementsComplementOrdinals.at(aExcludeNames);
        }
    }
            
    Plato::OrdinalVectorT<const Plato::OrdinalType>
    OmegaHMesh::GetSideSetFacesComplement( std::vector<std::string> aExcludeNames)
    {
        if( mSideSetFacesComplementOrdinals.count(aExcludeNames) )
        {
            return mSideSetFacesComplementOrdinals.at(aExcludeNames);
        }
        else
        {
            createComplement( aExcludeNames );
            return mSideSetFacesComplementOrdinals.at(aExcludeNames);
        }
    }
            
    Plato::OrdinalVectorT<const Plato::OrdinalType>
    OmegaHMesh::GetSideSetLocalNodesComplement( std::vector<std::string> aExcludeNames)
    {
        if( mSideSetLocalNodesComplementOrdinals.count(aExcludeNames) )
        {
            return mSideSetLocalNodesComplementOrdinals.at(aExcludeNames);
        }
        else
        {
            createComplement( aExcludeNames );
            return mSideSetLocalNodesComplementOrdinals.at(aExcludeNames);
        }
    }

    void
    OmegaHMesh::createComplement( std::vector<std::string> aExcludeNames )
    {
        if (mMesh.dim() == 3)
        {
            const int cSpaceDim = 3;
            const int cFaceDim = cSpaceDim-1;
            const int cNodesPerFace = cSpaceDim;

            auto tFaceOrds = Plato::omega_h::get_boundary_entities<cFaceDim, Omega_h::SIDE_SET>(aExcludeNames, mMesh, mMeshSets);

            auto tNumFaces = tFaceOrds.size();
            Plato::OrdinalVector tSideSetFaces("side set faces", tNumFaces);
            Plato::OrdinalVector tSideSetElements("side set elements", tNumFaces);
            Plato::OrdinalVector tSideSetLocalNodes("side set local nodes", cNodesPerFace*tNumFaces);

            this->InvertSideSet<cSpaceDim>(tFaceOrds, tSideSetFaces, tSideSetElements, tSideSetLocalNodes);

            mSideSetFacesComplementOrdinals[aExcludeNames] = tSideSetFaces;
            mSideSetElementsComplementOrdinals[aExcludeNames] = tSideSetElements;
            mSideSetLocalNodesComplementOrdinals[aExcludeNames] = tSideSetLocalNodes;
        }
        else
        if (mMesh.dim() == 2)
        {
            const int cSpaceDim = 2;
            const int cFaceDim = cSpaceDim-1;
            const int cNodesPerFace = cSpaceDim;

            auto tFaceOrds = Plato::omega_h::get_boundary_entities<cFaceDim, Omega_h::SIDE_SET>(aExcludeNames, mMesh, mMeshSets);

            auto tNumFaces = tFaceOrds.size();
            Plato::OrdinalVector tSideSetFaces("side set faces", tNumFaces);
            Plato::OrdinalVector tSideSetElements("side set elements", tNumFaces);
            Plato::OrdinalVector tSideSetLocalNodes("side set local nodes", cNodesPerFace*tNumFaces);

            this->InvertSideSet<cSpaceDim>(tFaceOrds, tSideSetFaces, tSideSetElements, tSideSetLocalNodes);

            mSideSetFacesComplementOrdinals[aExcludeNames] = tSideSetFaces;
            mSideSetElementsComplementOrdinals[aExcludeNames] = tSideSetElements;
            mSideSetLocalNodesComplementOrdinals[aExcludeNames] = tSideSetLocalNodes;
        }
        else
        if (mMesh.dim() == 1)
        {
            const int cSpaceDim = 1;
            const int cFaceDim = cSpaceDim-1;
            const int cNodesPerFace = cSpaceDim;

            auto tFaceOrds = Plato::omega_h::get_boundary_entities<cFaceDim, Omega_h::SIDE_SET>(aExcludeNames, mMesh, mMeshSets);

            auto tNumFaces = tFaceOrds.size();
            Plato::OrdinalVector tSideSetFaces("side set faces", tNumFaces);
            Plato::OrdinalVector tSideSetElements("side set elements", tNumFaces);
            Plato::OrdinalVector tSideSetLocalNodes("side set local nodes", cNodesPerFace*tNumFaces);

            this->InvertSideSet<cSpaceDim>(tFaceOrds, tSideSetFaces, tSideSetElements, tSideSetLocalNodes);

            mSideSetFacesComplementOrdinals[aExcludeNames] = tSideSetFaces;
            mSideSetElementsComplementOrdinals[aExcludeNames] = tSideSetElements;
            mSideSetLocalNodesComplementOrdinals[aExcludeNames] = tSideSetLocalNodes;
        }
        else
        {
            throw std::runtime_error("Unsupported mesh dimension");
        }
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    OmegaHMesh::GetNodeSetNodes( std::string aNodeSetName) const
    {
        auto& tNodeSets = mMeshSets[Omega_h::NODE_SET];

        // parse child nodes
        auto tNodeSetsIter = tNodeSets.find(aNodeSetName);
        if(tNodeSetsIter == tNodeSets.end())
        {
            std::ostringstream tMsg;
            tMsg << "Could not find Node Set with name = '" << aNodeSetName.c_str()
                    << "'. Node Set is not defined in input geometry/mesh file.\n";
            ANALYZE_THROWERR(tMsg.str())
        }
        return tNodeSetsIter->second.view();
    }
    
    void
    OmegaHMesh::CreateNodeSet(
        std::string                               aNodeSetName,
        std::initializer_list<Plato::OrdinalType> aNodes
    )
    {
        if( mMeshSets[Omega_h::NODE_SET].count(aNodeSetName) )
        {
            ANALYZE_THROWERR("Node set already exists.  Names must be unique.");
        }
        Omega_h::LOs tNodes(aNodes);
        mMeshSets[Omega_h::NODE_SET][aNodeSetName] = tNodes;
    }
}
