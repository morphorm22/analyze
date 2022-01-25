#pragma once

#include <memory>

#include "AbstractPlatoMesh.hpp"

#include <data_mesh.hpp>
#include <data_container.hpp>

namespace Plato
{
    class EngineMeshIO;
    class EngineMesh : public AbstractMesh
    {
        friend EngineMeshIO;

        std::string mFileName;
        std::shared_ptr<DataMesh> mMesh;
        std::shared_ptr<DataContainer> mDataContainer;

        Plato::OrdinalType mNumNodes;
        Plato::OrdinalType mNumElements;
        Plato::OrdinalType mNumDimensions;
        Plato::OrdinalType mNumNodesPerElement;

        Plato::OrdinalVector mConnectivity;
        Plato::ScalarVector mCoordinates;

        Plato::OrdinalVector mNodeElementGraph_offsets;
        Plato::OrdinalVector mNodeElementGraph_ordinals;

        Plato::OrdinalVector mNodeNodeGraph_offsets;
        Plato::OrdinalVector mNodeNodeGraph_ordinals;

        Plato::OrdinalMultiVector mFaceGraph;
        Plato::HostOrdinalMultiVector mFaceGraphHost;

        std::map<std::string, Plato::OrdinalVector> mBlockElementOrdinals;

        std::map<std::string, Plato::OrdinalVector> mSideSetFaceOrdinals;
        std::map<std::string, Plato::OrdinalVector> mSideSetElementOrdinals;
        std::map<std::string, Plato::OrdinalVector> mSideSetLocalNodeOrdinals;

        std::map<std::vector<std::string>, Plato::OrdinalVector> mSideSetFacesComplementOrdinals;
        std::map<std::vector<std::string>, Plato::OrdinalVector> mSideSetElementsComplementOrdinals;
        std::map<std::vector<std::string>, Plato::OrdinalVector> mSideSetLocalNodesComplementOrdinals;

        Plato::OrdinalVector mFullSurfaceSideSetFaceOrdinals;
        Plato::OrdinalVector mFullSurfaceSideSetElementOrdinals;
        Plato::OrdinalVector mFullSurfaceSideSetLocalNodeOrdinals;

        std::map<std::string, Plato::OrdinalVector> mNodeSetOrdinals;

        template<typename T1, typename T2>
        void copy(T1& tTo, T2 tFrom)
        {
          Kokkos::resize(tTo, tFrom.size());
          Kokkos::deep_copy(tTo, tFrom);
        }

        public:
            EngineMesh(std::string aInputMeshName);

            ~EngineMesh();

            std::string FileName() const override;
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

            void initialize();
            void openMesh();
            void closeMesh();
            void loadConnectivity();
            void loadCoordinates();
            void createNodeElementGraph();
            void createNodeNodeGraph();
            void createElementBlocks();
            void loadNodeSets();
            void loadSideSets();

            void createFullSurfaceSideSet();
    };
}
