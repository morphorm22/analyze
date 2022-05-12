/*
 * PlatoTestHelpers.hpp
 *
 *  Created on: Mar 31, 2018
 */

#ifndef PLATOTESTHELPERS_HPP_
#define PLATOTESTHELPERS_HPP_

#include <fstream>
#include <Teuchos_RCP.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "BamG.hpp"

#include "AnalyzeMacros.hpp"
#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"

namespace PlatoUtestHelpers
{
/******************************************************************************/
/*! Return a box (cube) mesh.
 * \param aMeshIntervals Number of mesh intervals through the thickness.
 * \param aMeshType Mesh type (i.e., TET4, hex8, TRI3, quad4, bar2, etc.)
 *
 * The mesh will have sidesets and nodesets on all faces, edges, and vertices
 * named 'x+' for the positive x face, 'x+y-' for the positive x negative y edge,
 * and 'x-y+z-' for the negative x positive y negative z vertex, etc.
 */
inline Plato::Mesh
 getBoxMesh(
    std::string        aMeshType,
    Plato::OrdinalType aMeshIntervals,
    std::string        aFileName = "BamG_unit_test_mesh.exo")
/******************************************************************************/
{
    BamG::MeshSpec tSpec;
    tSpec.meshType = aMeshType;
    tSpec.fileName = aFileName;
    tSpec.numX = aMeshIntervals;
    tSpec.numY = aMeshIntervals;
    tSpec.numZ = aMeshIntervals;

    BamG::generate(tSpec);

    return Plato::MeshFactory::create(tSpec.fileName);
}

/******************************************************************************/
/*! Return a box (cube) mesh.
 * \param aMeshType Mesh type (i.e., TET4, hex8, TRI3, quad4, bar2, etc.)
 * \param aMeshIntervalsX Number of mesh intervals in X, etc.
 * \param aMeshWidthX Width of mesh in X, etc.
 *
 * The mesh will have sidesets and nodesets on all faces, edges, and vertices
 * named 'x+' for the positive x face, 'x+y-' for the positive x negative y edge,
 * and 'x-y+z-' for the negative x positive y negative z vertex, etc.
 */
inline Plato::Mesh
 getBoxMesh(
    std::string        aMeshType,
    Plato::Scalar      aMeshWidthX,
    Plato::OrdinalType aMeshIntervalsX,
    Plato::Scalar      aMeshWidthY=1.0,
    Plato::OrdinalType aMeshIntervalsY=1,
    Plato::Scalar      aMeshWidthZ=1.0,
    Plato::OrdinalType aMeshIntervalsZ=1)
/******************************************************************************/
{
    BamG::MeshSpec tSpec;
    tSpec.meshType = aMeshType;
    tSpec.fileName = "BamG_unit_test_mesh.exo";
    tSpec.numX = aMeshIntervalsX;
    tSpec.numY = aMeshIntervalsY;
    tSpec.numZ = aMeshIntervalsZ;
    tSpec.dimX = aMeshWidthX;
    tSpec.dimY = aMeshWidthY;
    tSpec.dimZ = aMeshWidthZ;

    BamG::generate(tSpec);

    return Plato::MeshFactory::create(tSpec.fileName);
}

/******************************************************************************//**
 * \brief get view from device
 *
 * \param[in] aView data on device
 * \returns Mirror on host
**********************************************************************************/
template <typename ViewType>
typename ViewType::HostMirror
get(ViewType aView)
{
    using RetType = typename ViewType::HostMirror;
    RetType tView = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView, aView);
    return tView;
}

/******************************************************************************//**
 * \brief Set Dirichlet boundary condition values for specified degree of freedom.
 *   Specialized for 2-D applications
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofValues  vector of Dirichlet boundary condition values
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 * \param [in] aSetValue   value to set
 *
 **********************************************************************************/
inline void set_dof_value_in_vector_on_boundary_2D(Plato::Mesh aMesh,
                                                   const std::string & aBoundaryID,
                                                   const Plato::ScalarVector & aDofValues,
                                                   const Plato::OrdinalType & aDofStride,
                                                   const Plato::OrdinalType & aDofToSet,
                                                   const Plato::Scalar & aSetValue)
{
    auto tBoundaryNodes = aMesh->GetNodeSetNodes(aBoundaryID);

    auto tNumBoundaryNodes = tBoundaryNodes.size();

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBoundaryNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * tBoundaryNodes(aIndex) + aDofToSet;
        aDofValues(tIndex) += aSetValue;
    },
                         "fill vector boundary dofs");
}

/******************************************************************************//**
 * \brief Set Dirichlet boundary condition values for specified degree of freedom.
 *   Specialized for 3-D applications.
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofValues  vector of Dirichlet boundary condition values
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 * \param [in] aSetValue   value to set
 *
 **********************************************************************************/
inline void set_dof_value_in_vector_on_boundary_3D(Plato::Mesh aMesh,
                                                   const std::string & aBoundaryID,
                                                   const Plato::ScalarVector & aDofValues,
                                                   const Plato::OrdinalType & aDofStride,
                                                   const Plato::OrdinalType & aDofToSet,
                                                   const Plato::Scalar & aSetValue)
{
    auto tLocalOrdinals = aMesh->GetNodeSetNodes(aBoundaryID);
    auto tNumBoundaryNodes = tLocalOrdinals.size();

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBoundaryNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * tLocalOrdinals(aIndex) + aDofToSet;
        aDofValues(tIndex) += aSetValue;
    }, "fill vector boundary dofs");
}

/******************************************************************************//**
 * \brief Return list of Dirichlet degree of freedom indices, specialized for 2-D applications.
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 *
 * \return list of Dirichlet indices
 *
 **********************************************************************************/
inline Plato::OrdinalVector
get_dirichlet_indices_on_boundary_2D(
          Plato::Mesh          aMesh,
    const std::string        & aBoundaryID,
    const Plato::OrdinalType & aDofStride,
    const Plato::OrdinalType & aDofToSet
)
{
    auto tBoundaryNodes = aMesh->GetNodeSetNodes(aBoundaryID);
    Plato::OrdinalVector tDofIndices;
    auto tNumBoundaryNodes = tBoundaryNodes.size();
    Kokkos::resize(tDofIndices, tNumBoundaryNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBoundaryNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * tBoundaryNodes[aIndex] + aDofToSet;
        tDofIndices(aIndex) = tIndex;
    }, "fill dirichlet dof indices on boundary 2-D");

    return (tDofIndices);
}

/******************************************************************************//**
 * \brief Return list of Dirichlet degree of freedom indices, specialized for 3-D applications.
 *
 * \param [in]     aMesh       finite element mesh
 * \param [in]     aBoundaryID boundary identifier
 * \param [in]     aDofStride  degree of freedom stride
 * \param [in]     aDofToSet   degree of freedom index to set
 *
 * \return list of Dirichlet indices
 *
 **********************************************************************************/
inline Plato::OrdinalVector
get_dirichlet_indices_on_boundary_3D(
          Plato::Mesh          aMesh,
    const std::string        & aBoundaryID,
    const Plato::OrdinalType & aDofStride,
    const Plato::OrdinalType & aDofToSet)
{
    auto tBoundaryNodes = aMesh->GetNodeSetNodes(aBoundaryID);
    Plato::OrdinalVector tDofIndices;
    auto tNumBoundaryNodes = tBoundaryNodes.size();
    Kokkos::resize(tDofIndices, tNumBoundaryNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBoundaryNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * tBoundaryNodes[aIndex] + aDofToSet;
        tDofIndices(aIndex) = tIndex;
    }, "fill dirichlet dof indices on boundary 3-D");

    return (tDofIndices);
}

/******************************************************************************//**
 * \brief set value for this Dirichlet boundary condition index
 *
 * \param [in] aDofValues vector of Dirichlet boundary condition values
 * \param [in] aDofStride degree of freedom stride
 * \param [in] aDofToSet  degree of freedom index to set
 * \param [in] aSetValue  value to set
 *
 **********************************************************************************/
inline void set_dof_value_in_vector(const Plato::ScalarVector & aDofValues,
                                    const Plato::OrdinalType & aDofStride,
                                    const Plato::OrdinalType & aDofToSet,
                                    const Plato::Scalar & aSetValue)
{
    auto tVectorSize = aDofValues.extent(0);
    auto tRange = tVectorSize / aDofStride;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * aNodeIndex + aDofToSet;
        aDofValues(tIndex) += aSetValue;
    }, "fill specific vector entry globally");
}

inline std::vector<std::vector<Plato::Scalar>>
toFull( Teuchos::RCP<Plato::CrsMatrixType> aInMatrix )
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    std::vector<std::vector<Scalar>>
        retMatrix(aInMatrix->numRows(),std::vector<Scalar>(aInMatrix->numCols(),0.0));

    auto tNumRowsPerBlock = aInMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aInMatrix->numColsPerBlock();
    auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

    auto tRowMap = get(aInMatrix->rowMap());
    auto tColMap = get(aInMatrix->columnIndices());
    auto tValues = get(aInMatrix->entries());

    auto tNumRows = tRowMap.extent(0)-1;
    for(OrdinalType iRowIndex=0; iRowIndex<tNumRows; iRowIndex++)
    {
        auto tFrom = tRowMap(iRowIndex);
        auto tTo   = tRowMap(iRowIndex+1);
        for(auto iColMapEntryIndex=tFrom; iColMapEntryIndex<tTo; iColMapEntryIndex++)
        {
            auto tBlockColIndex = tColMap(iColMapEntryIndex);
            for(OrdinalType iLocalRowIndex=0; iLocalRowIndex<tNumRowsPerBlock; iLocalRowIndex++)
            {
                auto tRowIndex = iRowIndex * tNumRowsPerBlock + iLocalRowIndex;
                for(OrdinalType iLocalColIndex=0; iLocalColIndex<tNumColsPerBlock; iLocalColIndex++)
                {
                    auto tColIndex = tBlockColIndex * tNumColsPerBlock + iLocalColIndex;
                    auto tSparseIndex = iColMapEntryIndex * tBlockSize + iLocalRowIndex * tNumColsPerBlock + iLocalColIndex;
                    retMatrix[tRowIndex][tColIndex] = tValues[tSparseIndex];
                }
            }
        }
    }
    return retMatrix;
}

/******************************************************************************//**
 * \brief ignore a variable and suppress compiler warnings :)
 *
 * \tparam [in] Any typename
 **********************************************************************************/
template <typename T>
void ignore_unused_variable_warning(T &&) {}

} // namespace PlatoUtestHelpers

#endif /* PLATOTESTHELPERS_HPP_ */
