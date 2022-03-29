#pragma once

#include "AbstractPlatoMesh.hpp"

#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_assoc.hpp>

namespace Plato
{
    namespace OmegaH
    {
        extern Omega_h::Library* Library;
    }
    class OmegaHMesh : public AbstractMesh
    {
        const std::string mFileName;
        std::string mElementType;
        Omega_h::Assoc mAssoc;

        std::map<std::string, Plato::OrdinalVector> mSideSetFacesOrdinals;
        std::map<std::string, Plato::OrdinalVector> mSideSetElementsOrdinals;
        std::map<std::string, Plato::OrdinalVector> mSideSetLocalNodesOrdinals;

        std::map<std::vector<std::string>, Plato::OrdinalVector> mSideSetFacesComplementOrdinals;
        std::map<std::vector<std::string>, Plato::OrdinalVector> mSideSetElementsComplementOrdinals;
        std::map<std::vector<std::string>, Plato::OrdinalVector> mSideSetLocalNodesComplementOrdinals;

        void initialize();

        public: // temporarily
        Omega_h::Mesh mMesh;
        Omega_h::MeshSets mMeshSets;

        public:
            OmegaHMesh(std::string aInputMeshName);
            OmegaHMesh(Omega_h::Mesh& aMesh) : mMesh(aMesh) {initialize();}
            OmegaHMesh(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets) : mMesh(aMesh), mMeshSets(aMeshSets) {initialize();}

            OmegaHMesh(std::shared_ptr<Plato::OmegaHMesh> aOmegaHMesh) : OmegaHMesh(*aOmegaHMesh) {}

            std::string FileName() const override;
            std::string ElementType() const override;
            Plato::OrdinalType NumNodes() const override;
            Plato::OrdinalType NumElements() const override;
            Plato::OrdinalType NumNodesPerElement() const override;
            Plato::OrdinalType NumDimensions() const override;

            Plato::ScalarVectorT<const Plato::OrdinalType>
            GetLocalElementIDs(std::string aBlockName) const override;

            std::vector<std::string> GetElementBlockNames() const override;
            std::vector<std::string> GetNodeSetNames() const override;
            std::vector<std::string> GetSideSetNames() const override;

            Plato::ScalarVectorT<const Plato::Scalar>
            Coordinates() const override;

            void SetCoordinates(Plato::ScalarVector) override;

            Plato::OrdinalVectorT<const Plato::OrdinalType>
            Connectivity() override;

            void
            NodeNodeGraph(
                Plato::OrdinalVectorT<const Plato::OrdinalType> & aOffsetMap,
                Plato::OrdinalVectorT<const Plato::OrdinalType> & aNodeOrds
            ) override;

            void
            NodeElementGraph(
                Plato::OrdinalVectorT<const Plato::OrdinalType> & aOffsetMap,
                Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds
            ) override;

            
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetFaces( std::string aSideSetName) const override;
            
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetElements( std::string aSideSetName) const override;
            
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetLocalNodes( std::string aSideSetName) const override;
            
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetElementsComplement( std::vector<std::string> aExcludeNames) override;
            
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetFacesComplement( std::vector<std::string> aExcludeNames) override;
            
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetLocalNodesComplement( std::vector<std::string> aExcludeNames) override;

            void
            createComplement( std::vector<std::string> aExcludeNames );
            
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetNodeSetNodes( std::string aNodeSetName) const override;

            void
            CreateNodeSet( std::string aNodeSetName, std::initializer_list<Plato::OrdinalType> aNodes) override;

            template<int cSpaceDims>
            void
            InvertSideSet(
                const Omega_h::LOs         & aFaceLids,
                      Plato::OrdinalVector & aSideSetFaces,
                      Plato::OrdinalVector & aSideSetElements,
                      Plato::OrdinalVector & aSideSetLocalNodes
            )
            {
                // get mesh vertices
                const auto cNodesPerFace = cSpaceDims;
                const auto cNodesPerCell = cSpaceDims+1;
                const auto cNumSpaceDimsOnFace = cSpaceDims - 1;
                const auto cNumFacesPerCell = cSpaceDims + 1;
                auto tFace2Verts = mMesh.ask_verts_of(cNumSpaceDimsOnFace);
                auto tCell2Verts = mMesh.ask_elem_verts();

                auto tFace2eElems = mMesh.ask_up(cNumSpaceDimsOnFace, cSpaceDims);
                auto tFace2Elems_map   = tFace2eElems.a2ab;
                auto tFace2Elems_elems = tFace2eElems.ab2b;

                auto tElem2Faces = mMesh.ask_down(cSpaceDims, cNumSpaceDimsOnFace).ab2b;

                auto tNumFaces = aFaceLids.size();
                Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
                {
                    auto tFaceOrdinal = aFaceLids[aFaceI];
                    for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal+1]; ++tElem )
                    {
                        auto tCellOrdinal = tFace2Elems_elems[tElem];
                        aSideSetElements(aFaceI) = tCellOrdinal;

                        for(Plato::OrdinalType tFace = 0; tFace < cNumFacesPerCell; tFace++)
                        {
                            if(tElem2Faces[tCellOrdinal*cNumFacesPerCell+tFace] == tFaceOrdinal)
                            {
                                aSideSetFaces(aFaceI) = tFace;
                            }
                        }

                        for( Plato::OrdinalType tNodeI = 0; tNodeI < cNodesPerFace; tNodeI++)
                        {
                            for( Plato::OrdinalType tNodeJ = 0; tNodeJ < cNodesPerCell; tNodeJ++)
                            {
                                if( tFace2Verts[tFaceOrdinal*cNodesPerFace+tNodeI] == tCell2Verts[tCellOrdinal*cNodesPerCell + tNodeJ] )
                                {
                                    aSideSetLocalNodes(aFaceI*cNodesPerFace+tNodeI) = tNodeJ;
                                }
                            }
                        }
                    }
                }, "invert sideset");
            }

            template<int cSpaceDims>
            void
            InvertSideSets()
            {
                const auto cNodesPerFace = cSpaceDims;

                for( auto& tSideSetPair : mMeshSets[Omega_h::SIDE_SET] )
                {
                    auto tSideSetName = tSideSetPair.first;

                    auto tFaceLids = tSideSetPair.second;

                    auto tNumFaces = tFaceLids.size();

                    Plato::OrdinalVector tSideSetFaces("side set faces", tNumFaces);
                    Plato::OrdinalVector tSideSetElements("side set elements", tNumFaces);
                    Plato::OrdinalVector tSideSetLocalNodes("side set local nodes", cNodesPerFace*tNumFaces);

                    this->InvertSideSet<cSpaceDims>(tFaceLids, tSideSetFaces, tSideSetElements, tSideSetLocalNodes);

                    mSideSetFacesOrdinals[tSideSetName] = tSideSetFaces;
                    mSideSetElementsOrdinals[tSideSetName] = tSideSetElements;
                    mSideSetLocalNodesOrdinals[tSideSetName] = tSideSetLocalNodes;
                }
            }

    };
}
