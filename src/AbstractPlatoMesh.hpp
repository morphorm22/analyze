#pragma once

#include <initializer_list>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

    class AbstractMesh
    {
        public:
            /******************************************************************************//**
            * \brief Return the name of the mesh file
            **********************************************************************************/
            virtual std::string FileName() const = 0;

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
            * \brief Return number of nodes per element in the mesh
            **********************************************************************************/
            virtual Plato::OrdinalType NumNodesPerElement() const = 0;

            /******************************************************************************//**
            * \brief Return number of dimensions
            **********************************************************************************/
            virtual Plato::OrdinalType NumDimensions() const = 0;

            /******************************************************************************//**
            * \brief Returns rank-local element ids for the given block name
            * \returns const Plato::OrdinalVector of element ids
            **********************************************************************************/
            virtual Plato::ScalarVectorT<const Plato::OrdinalType>
            GetLocalElementIDs(std::string aBlockName) const = 0;

            /******************************************************************************//**
            * \brief Returns the names of all blocks in the mesh
            * \returns std::vector of block names
            **********************************************************************************/
            virtual std::vector<std::string>
            GetElementBlockNames() const = 0;
        
            /******************************************************************************//**
            * \brief Returns the names of all nodesets in the mesh
            * \returns std::vector of nodeset names
            **********************************************************************************/
            virtual std::vector<std::string>
            GetNodeSetNames() const = 0;
        
            /******************************************************************************//**
            * \brief Returns the names of all sidesets in the mesh
            * \returns std::vector of sideset names
            **********************************************************************************/
            virtual std::vector<std::string>
            GetSideSetNames() const = 0;
        
            /******************************************************************************//**
            * \brief Returns coordinates for all nodes
            * \returns Plato::ScalarVector of coordinates: {x0, y0, z0, x1, y1, z1, ..., zN}
            **********************************************************************************/
            virtual Plato::ScalarVectorT<const Plato::Scalar> Coordinates() const = 0;
        
            /******************************************************************************//**
            * \brief Set coordinates for all nodes
            * \returns Plato::ScalarVector of coordinates: {x0, y0, z0, x1, y1, z1, ..., zN}
            **********************************************************************************/
            virtual void SetCoordinates(Plato::ScalarVector) = 0;
        
            /******************************************************************************//**
            * \brief Returns connectvity for all elements
            * \returns Plato::OrdinalVector of connectivity:
            *   {N_{0,0}, N_{0,1}, ..., N_{0,n-1}, N_{1,0}, N_{1,1}, ..., N_{1,n}, ..., N_{e-1,n-1}}
            * where n is the number of nodes per element and e is the number of elements
            **********************************************************************************/
            virtual Plato::OrdinalVectorT<const Plato::OrdinalType> Connectivity() = 0;
        
            /******************************************************************************//**
            * \brief Get node-to-node graph
            * \param [out] aOffsetMap offsets into aNodeOrds.  aOffsetMap(NodeIndex) and 
            *        aOffsetMap(NodeIndex+1) return indices of the first and last entries in 
            *        aNodeOrds for node with local index, NodeIndex.
            * \param [out] aNodeOrds local node indices 
            **********************************************************************************/
            virtual void
            NodeNodeGraph(
                Plato::OrdinalVectorT<const Plato::OrdinalType> & aOffsetMap,
                Plato::OrdinalVectorT<const Plato::OrdinalType> & aNodeOrds) = 0;
        
            /******************************************************************************//**
            * \brief Get node-to-element graph
            * \param [out] aOffsetMap offsets into aElementOrds.  aOffsetMap(NodeIndex) and 
            *        aOffsetMap(NodeIndex+1) return indices of the first and last entries in 
            *        aElementOrds for node with local index, NodeIndex.
            * \param [out] aElementOrds local node indices 
            **********************************************************************************/
            virtual void
            NodeElementGraph(
                Plato::OrdinalVectorT<const Plato::OrdinalType> & aOffsetMap,
                Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds) = 0;
        
            /******************************************************************************//**
            * \brief Get node set node ordinals 
            * \param [in] aNodeSetName name of the node set
            * \returns const OrdinalVector of mesh-local node indices for the named nodeset
            * If the mesh is distributed, returns only the nodes on the local partition.
            **********************************************************************************/
            virtual
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetNodeSetNodes( std::string aNodeSetName) const = 0;
        
            /******************************************************************************//**
            * \brief Create node set from node ordinals 
            * \param [in] aNodeSetName name of the node set to be created
            * \param [in] aNodes nodes to be added to the new node set
            * \returns const OrdinalVector of mesh-local node indices for the named nodeset
            * If the mesh is distributed, returns only the nodes on the local partition.
            **********************************************************************************/
            virtual
            void
            CreateNodeSet( std::string aNodeSetName, std::initializer_list<Plato::OrdinalType> aNodes) = 0;
        
            /******************************************************************************//**
            * \brief Get side set face element-local ordinals (i.e., numbered 0 through n-1)
            * where n is the number of faces per element.
            * \param [in] aSideSetName name of the side set
            * \returns const OrdinalVector of edges (2D) or faces (3D) for the named side set
            **********************************************************************************/
            virtual
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetFaces( std::string aSideSetName) const = 0;

            /******************************************************************************//**
            * \brief Get side set rank-local element ordinals 
            * \param [in] aSideSetName name of the side set
            * \returns const OrdinalVector of element ordinals for the named side set
            **********************************************************************************/
            virtual
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetElements( std::string aSideSetName) const = 0;

            /******************************************************************************//**
            * \brief Get side set element-local node ordinals (i.e., numbered 0 through n-1)
            * where n is the number of nodes per element.
            * \param [in] aSideSetName name of the side set
            * \returns const OrdinalVector of node ordinals for the named side set
            *   {N_{0,0}, N_{0,1}, ..., N_{0,m-1}, N_{1,0}, N_{1,1}, ..., N_{1,m-1}, ..., N_{f-1,m-1}}
            * where m is the number of nodes per face, and f is the number of faces in the side set
            **********************************************************************************/
            virtual
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetLocalNodes( std::string aSideSetName) const = 0;

            /******************************************************************************//**
            * \brief Get side set rank-local element ordinals for side set complement
            * \param [in] aExcludeNames names of side sets to be excluded from the complement
            * \returns const OrdinalVector of element ordinals
            **********************************************************************************/
            virtual
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetElementsComplement( std::vector<std::string> aExcludeNames) = 0;

            /******************************************************************************//**
            * \brief Get side set face element-local ordinals (i.e., numbered 0 through n-1)
            * where n is the number of faces per element.
            * \param [in] aSideSetName name of the side set
            * \returns const OrdinalVector of edges (2D) or faces (3D) for the named side set
            **********************************************************************************/
            virtual
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetFacesComplement( std::vector<std::string> aSideSetName) = 0;

            /******************************************************************************//**
            * \brief Get side set element-local node ordinals (i.e., numbered 0 through n-1) for
            * the side set complement, where n is the number of nodes per element.
            * \param [in] aExcludeNames names of the side sets to be excluded from the complement
            * \returns const OrdinalVector of node ordinals.
            *   {N_{0,0}, N_{0,1}, ..., N_{0,m-1}, N_{1,0}, N_{1,1}, ..., N_{1,m-1}, ..., N_{f-1,m-1}}
            * where m is the number of nodes per face, and f is the number of faces in the side set
            **********************************************************************************/
            virtual
            Plato::OrdinalVectorT<const Plato::OrdinalType>
            GetSideSetLocalNodesComplement( std::vector<std::string> aExcludeNames) = 0;
        
    };
}
