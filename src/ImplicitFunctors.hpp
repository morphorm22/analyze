#ifndef PLATO_IMPLICIT_FUNCTORS_HPP
#define PLATO_IMPLICIT_FUNCTORS_HPP

#include "PlatoMesh.hpp"
#include "PlatoMathTypes.hpp"

#include <Teuchos_RCPDecl.hpp>

namespace Plato
{

/******************************************************************************//**
* \brief functor that provides mesh-local node ordinal
* \param [in] aMesh Plato abstract mesh
**********************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NodesPerCell=SpaceDim+1>
class NodeOrdinal
{
  public:
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    NodeOrdinal(
      Plato::Mesh aMesh ) :
      mCells2nodes(aMesh->Connectivity()) {}

    /******************************************************************************//**
    * \brief Returns mesh-local node ordinal
    * \param [in] aCellOrdinal mesh-local element ordinal
    * \param [in] aNodeOrdinal elem-local node ordinal
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::OrdinalType aNodeOrdinal
    ) const
    {
        return mCells2nodes(aCellOrdinal*NodesPerCell + aNodeOrdinal);
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode, Plato::OrdinalType NodesPerCell=SpaceDim+1>
class VectorEntryOrdinal
{
  public:
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    VectorEntryOrdinal(
      Plato::Mesh mesh ) :
      mCells2nodes(mesh->Connectivity()) {}

    KOKKOS_INLINE_FUNCTION Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType nodeOrdinal, Plato::OrdinalType dofOrdinal=0) const
    {
        Plato::OrdinalType vertexNumber = mCells2nodes(cellOrdinal*NodesPerCell + nodeOrdinal);
        return vertexNumber * DofsPerNode + dofOrdinal;
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NodesPerCell>
class NodeCoordinate
{
  private:
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;
    const Plato::ScalarVectorT<const Plato::Scalar> mCoords;

  public:
    NodeCoordinate(Plato::Mesh aMesh) :
      mCells2nodes(aMesh->Connectivity()),
      mCoords(aMesh->Coordinates())
      {
        if (aMesh->NumDimensions() != SpaceDim || aMesh->NumNodesPerElement() != NodesPerCell)
        {
            throw std::runtime_error("Input mesh doesn't match physics spatial dimension and/or nodes per cell.");
        }
      }

    KOKKOS_INLINE_FUNCTION
    Plato::Scalar
    operator()(Plato::OrdinalType aCellOrdinal, Plato::OrdinalType aNodeOrdinal, Plato::OrdinalType aDimOrdinal) const
    {
        const Plato::OrdinalType tVertexNumber = mCells2nodes(aCellOrdinal*NodesPerCell + aNodeOrdinal);
        const Plato::Scalar tCoord = mCoords(tVertexNumber * SpaceDim + aDimOrdinal);
        return tCoord;
    }
};
/******************************************************************************/

/******************************************************************************/
/*! InertialForces Functor.
*
*   Evaluates cell inertial forces.
*/
/******************************************************************************/
class ComputeProjectionWorkset
{
public:
    /******************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarVectorT<Plato::Scalar> & tBasisFunctions,
                                       const Plato::ScalarMultiVectorT<GaussPointScalarType> & aStateValues,
                                       const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult,
                                             Plato::Scalar scale = 1.0 ) const
    /******************************************************************************/
    {
        const Plato::OrdinalType tNumNodesPerCell = tBasisFunctions.size();
        const Plato::OrdinalType tNumDofsPerNode = aStateValues.extent(1);
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
            {
                Plato::OrdinalType tMyDofIndex = (tNumDofsPerNode * tNodeIndex) + tDofIndex;
                aResult(aCellOrdinal, tMyDofIndex) += scale * tBasisFunctions(tNodeIndex)
                        * aStateValues(aCellOrdinal, tDofIndex) * aCellVolume(aCellOrdinal);
            }
        }
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType NodesPerCell,
         Plato::OrdinalType DofsPerNode>
class BlockMatrixTransposeEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVectorT mRowMap;
    const typename CrsMatrixType::OrdinalVectorT mColumnIndices;
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    BlockMatrixTransposeEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> matrix, Plato::Mesh mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->Connectivity()) { }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode;
        auto iDof  = icellDof % DofsPerNode;
        auto jNode = jcellDof / DofsPerNode;
        auto jDof  = jcellDof % DofsPerNode;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes(cellOrdinal * NodesPerCell + iNode);
        Plato::OrdinalType jLocalOrdinal = mCells2nodes(cellOrdinal * NodesPerCell + jNode);
        Plato::OrdinalType rowStart = mRowMap(jLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(jLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == iLocalOrdinal)
          {
            return entryOrdinal*DofsPerNode*DofsPerNode+jDof*DofsPerNode+iDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType BlockSize_I, Plato::OrdinalType BlockSize_J>
class LocalByGlobalEntryFunctor
{
  private:
    const typename CrsMatrixType::RowMapVectorT mRowMap;
    const typename CrsMatrixType::OrdinalVectorT mColumnIndices;
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

    using MatrixT = Teuchos::RCP<Plato::CrsMatrixType>;
    using MeshT   = Plato::Mesh;

  public:
    LocalByGlobalEntryFunctor(
        MatrixT tMatrix,
        MeshT   tMesh
    ) :
      mRowMap        (tMatrix->rowMap()),
      mColumnIndices (tMatrix->columnIndices()),
      mCells2nodes   (tMesh->Connectivity()) { }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto jNode = jcellDof / BlockSize_J;
        auto jDof  = jcellDof % BlockSize_J;
        Plato::OrdinalType jLocalOrdinal = mCells2nodes(cellOrdinal * (SpaceDim+1) + jNode);
        Plato::OrdinalType rowStart = mRowMap(cellOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(cellOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*BlockSize_I*BlockSize_J + icellDof*BlockSize_J + jDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};

/******************************************************************************/
//template<Plato::OrdinalType mNumNodesPerCell, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofsPerGP>
template<typename ElementType>
class GlobalByLocalEntryFunctor
{
    const typename CrsMatrixType::RowMapVectorT mRowMap;
    const typename CrsMatrixType::OrdinalVectorT mColumnIndices;
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    GlobalByLocalEntryFunctor(Teuchos::RCP<Plato::CrsMatrixType> matrix, Plato::Mesh mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->Connectivity()) { }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()(
        Plato::OrdinalType cellOrdinal,
        Plato::OrdinalType gpOrdinal,
        Plato::OrdinalType icellDof,
        Plato::OrdinalType jcellDof
    ) const
    {
        auto iNode = icellDof / ElementType::mNumDofsPerNode;
        auto iDof  = icellDof % ElementType::mNumDofsPerNode;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes(cellOrdinal * ElementType::mNumNodesPerCell + iNode);
        Plato::OrdinalType jLocalOrdinal = cellOrdinal*ElementType::mNumGaussPoints + gpOrdinal;
        Plato::OrdinalType rowStart = mRowMap(iLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(iLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*ElementType::mNumDofsPerNode*ElementType::mNumLocalStatesPerGP+iDof*ElementType::mNumLocalStatesPerGP+jcellDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};

/******************************************************************************/
template<Plato::OrdinalType NodesPerCell,
         Plato::OrdinalType DofsPerNode_I,
         Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
class BlockMatrixEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVectorT mRowMap;
    const typename CrsMatrixType::OrdinalVectorT mColumnIndices;
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    BlockMatrixEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> matrix, Plato::Mesh mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->Connectivity()) { }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode_I;
        auto iDof  = icellDof % DofsPerNode_I;
        auto jNode = jcellDof / DofsPerNode_J;
        auto jDof  = jcellDof % DofsPerNode_J;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes(cellOrdinal * NodesPerCell + iNode);
        Plato::OrdinalType jLocalOrdinal = mCells2nodes(cellOrdinal * NodesPerCell + jNode);
        Plato::OrdinalType rowStart = mRowMap(iLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(iLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*DofsPerNode_I*DofsPerNode_J+iDof*DofsPerNode_J+jDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};
/******************************************************************************/

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param aMesh Plato abstract mesh on which the matrix is based.

  Create a block matrix from connectivity in mesh with block size
  DofsPerNode_I X DofsPerElem_J.

  This function creates a matrix that stores a transpose of the gradient of
  local element states wrt nodal degrees of freedom.  Each column has the same
  number of non-zero block entries (NNodesPerCell)
*/
template <typename MatrixType,
          typename ElementType>
Teuchos::RCP<MatrixType>
CreateGlobalByLocalBlockMatrix( Plato::Mesh aMesh )
/******************************************************************************/
{
    Plato::OrdinalVectorT<const Plato::OrdinalType> tOffsetMap;
    Plato::OrdinalVectorT<const Plato::OrdinalType> tElementOrds;
    aMesh->NodeElementGraph(tOffsetMap, tElementOrds);

    auto tNumElems = aMesh->NumElements();
    auto tNumNodes = aMesh->NumNodes();
    auto tNumNonZeros = tNumElems*ElementType::mNumGaussPoints*ElementType::mNumNodesPerCell;

    constexpr Plato::OrdinalType numBlockDofs = ElementType::mNumDofsPerNode*ElementType::mNumLocalStatesPerGP;

    typename MatrixType::RowMapVectorT  rowMap        ("row map",        tNumNodes+1);
    typename MatrixType::ScalarVectorT  entries       ("matrix entries", tNumNonZeros*numBlockDofs);
    typename MatrixType::OrdinalVectorT columnIndices ("column indices", tNumNonZeros);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
    {
      auto tFrom = tOffsetMap(aNodeOrdinal);
      auto tTo   = tOffsetMap(aNodeOrdinal+1);
      rowMap(aNodeOrdinal)   = ElementType::mNumGaussPoints*tFrom;
      rowMap(aNodeOrdinal+1) = ElementType::mNumGaussPoints*tTo;

      for( decltype(tFrom) tOffset = tFrom; tOffset < tTo; tOffset++ )
      {
          for( decltype(tFrom) tGPOrd = 0; tGPOrd < ElementType::mNumGaussPoints; tGPOrd++ )
          {
              auto tColumnEntry = ElementType::mNumGaussPoints * tOffset + tGPOrd;
              columnIndices(tColumnEntry) = ElementType::mNumGaussPoints*tElementOrds(tOffset) + tGPOrd;
          }
      }
    });

    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries,
                     tNumNodes*ElementType::mNumDofsPerNode,
                     tNumElems*ElementType::mNumGaussPoints*ElementType::mNumLocalStatesPerGP,
                     ElementType::mNumDofsPerNode,
                     ElementType::mNumLocalStatesPerGP )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param aMesh Plato abstract mesh on which the matrix is based.

  Create a block matrix from connectivity in mesh with block size
  DofsPerElem_I X DofsPerNode_J.

  This function creates a matrix that stores a gradient of local element
  states wrt nodal degrees of freedom.  Each row has the same number of
  non-zero block entries (NNodesPerCell)
*/
template <typename MatrixType,
          Plato::OrdinalType NodesPerElem,
          Plato::OrdinalType DofsPerElem_I,
          Plato::OrdinalType DofsPerNode_J>
Teuchos::RCP<MatrixType>
CreateLocalByGlobalBlockMatrix( Plato::Mesh aMesh )
/******************************************************************************/
{
    const auto& mCells2nodes = aMesh->Connectivity();

    auto tNumElems = aMesh->NumElements();
    auto tNumNonZeros = tNumElems*NodesPerElem;

    constexpr Plato::OrdinalType numBlockDofs = DofsPerElem_I*DofsPerNode_J;

    typename MatrixType::RowMapVectorT  rowMap        ("row map",        tNumElems+1);
    typename MatrixType::ScalarVectorT  entries       ("matrix entries", tNumNonZeros*numBlockDofs);
    typename MatrixType::OrdinalVectorT columnIndices ("column indices", tNumNonZeros);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumElems), KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
      auto tFrom = aCellOrdinal*NodesPerElem;
      auto tTo   = (aCellOrdinal+1)*NodesPerElem;
      rowMap(aCellOrdinal)   = tFrom;
      rowMap(aCellOrdinal+1) = tTo;

      decltype(aCellOrdinal) tLocalIndex = 0;
      for( decltype(tFrom) tColumnEntry = tFrom; tColumnEntry < tTo; tColumnEntry++ )
      {
          columnIndices(tColumnEntry) = mCells2nodes(aCellOrdinal*NodesPerElem + tLocalIndex++);
      }
    });

    auto tNumNodes = aMesh->NumNodes();
    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries,
                     tNumElems*DofsPerElem_I, tNumNodes*DofsPerNode_J,
                     DofsPerElem_I, DofsPerNode_J )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  \param mesh Plato abstract mesh on which the matrix is based.  

  Create a block matrix from connectivity in mesh with block size
  DofsPerNode_I X DofsPerNode_J.
*/
template <typename MatrixType, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
Teuchos::RCP<MatrixType>
CreateBlockMatrix( Plato::Mesh aMesh )
/******************************************************************************/
{
    Plato::OrdinalVectorT<const Plato::OrdinalType> tOffsetMap;
    Plato::OrdinalVectorT<const Plato::OrdinalType> tNodeOrds;
    aMesh->NodeNodeGraph(tOffsetMap, tNodeOrds);

    // TODO: this function is still omega_h specific because it assumes that the graph doesn't include diagonals.

    auto numRows = tOffsetMap.size() - 1;
    // omega_h does not include the diagonals: add numRows, and then
    // add 1 to each rowMap entry after the first
    auto nnz = tNodeOrds.size() + numRows;

    // account for num dofs per node
    constexpr Plato::OrdinalType numBlockDofs = DofsPerNode_I*DofsPerNode_J;

    typename MatrixType::RowMapVectorT  rowMap("row map", numRows+1);
    typename MatrixType::ScalarVectorT  entries("matrix entries", nnz*numBlockDofs);
    typename MatrixType::OrdinalVectorT columnIndices("column indices", nnz);

    // The compressed row storage format in omega_h doesn't include diagonals.  This
    // function creates a CRSMatrix with diagonal entries included.

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numRows), KOKKOS_LAMBDA(Plato::OrdinalType rowNumber)
    {
      auto entryOffset_oh = tOffsetMap(rowNumber);
      auto R0 = tOffsetMap(rowNumber) + rowNumber;
      auto R1 = tOffsetMap(rowNumber+1) + rowNumber+1;
      auto numNodesThisRow = R1-R0;
      rowMap(rowNumber) = R0;
      rowMap(rowNumber+1) = R1;

      Plato::OrdinalType i_oh = 0; // will track i until we insert the diagonal entry
      for (Plato::OrdinalType i=0; i<numNodesThisRow; i_oh++, i++)
      {
        bool insertDiagonal = false;
        if ((i_oh == i) && (i_oh + entryOffset_oh >= tOffsetMap(rowNumber+1)))
        {
          // i_oh == i                    --> have not inserted diagonal
          // i_oh + entryOffset_oh > size --> at the end of the omega_h entries, should insert
          insertDiagonal = true;
        }
        else if (i_oh == i)
        {
          // i_oh + entryOffset_oh in bounds
          auto columnIndex = tNodeOrds(i_oh + entryOffset_oh);
          if (columnIndex > rowNumber)
          {
            insertDiagonal = true;
          }
        }
        if (insertDiagonal)
        {
          // store the diagonal entry
          columnIndices(R0+i) = rowNumber;
          i_oh--; // i_oh lags i by 1 after we hit the diagonal
        }
        else
        {
          columnIndices(R0+i) = tNodeOrds(i_oh + entryOffset_oh);
        }
      }
    });

    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries,
                     numRows*DofsPerNode_I, numRows*DofsPerNode_J,
                     DofsPerNode_I, DofsPerNode_J )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  \param aMesh Plato abstract mesh on which the matrix is based.  

  Create a matrix from connectivity in mesh with DofsPerNode.
*/
template <typename MatrixType, Plato::OrdinalType DofsPerNode>
Teuchos::RCP<MatrixType>
CreateMatrix( Plato::Mesh aMesh )
/******************************************************************************/
{
    Plato::OrdinalVectorT<const Plato::OrdinalType> tOffsetMap;
    Plato::OrdinalVectorT<const Plato::OrdinalType> tNodeOrds;
    aMesh->NodeNodeGraph(tOffsetMap, tNodeOrds);

    // TODO: this function is still omega_h specific because it assumes that the graph doesn't include diagonals.

    auto numRows = tOffsetMap.size() - 1;
    // omega_h does not include the diagonals: add numRows, and then
    // add 1 to each rowMap entry after the first
    auto nnz = tNodeOrds.size() + numRows;

    // account for num dofs per node
    constexpr Plato::OrdinalType numDofsSquared = DofsPerNode*DofsPerNode;

    typename MatrixType::RowMapVectorT  rowMap("row map", numRows*DofsPerNode+1);
    typename MatrixType::ScalarVectorT  entries("matrix entries", nnz*numDofsSquared);
    typename MatrixType::OrdinalVectorT columnIndices("column indices", nnz*numDofsSquared);

    // The compressed row storage format in omega_h doesn't include diagonals.  This
    // function creates a CRSMatrix with diagonal entries included and expands the
    // graph to DofsPerNode.

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numRows), KOKKOS_LAMBDA(Plato::OrdinalType rowNumber)
    {
      auto entryOffset_oh = tOffsetMap(rowNumber);
      auto R0 = tOffsetMap(rowNumber) + rowNumber;
      auto R1 = tOffsetMap(rowNumber+1) + rowNumber+1;
      auto numNodesThisRow = R1-R0;
      auto numDofsThisRow = numNodesThisRow*DofsPerNode;
      auto dofRowOffset = DofsPerNode*rowNumber;
      auto dofColOffset = numDofsSquared*R0;
      for (Plato::OrdinalType iDof=0; iDof<=DofsPerNode; iDof++){
        rowMap(dofRowOffset+iDof) = dofColOffset+iDof*numDofsThisRow;
      }

      Plato::OrdinalType i_oh = 0; // will track i until we insert the diagonal entry
      for (Plato::OrdinalType i=0; i<numNodesThisRow; i_oh++, i++)
      {
        bool insertDiagonal = false;
        if ((i_oh == i) && (i_oh + entryOffset_oh >= tOffsetMap(rowNumber+1)))
        {
          // i_oh == i                    --> have not inserted diagonal
          // i_oh + entryOffset_oh > size --> at the end of the omega_h entries, should insert
          insertDiagonal = true;
        }
        else if (i_oh == i)
        {
          // i_oh + entryOffset_oh in bounds
          auto columnIndex = tNodeOrds(i_oh + entryOffset_oh);
          if (columnIndex > rowNumber)
          {
            insertDiagonal = true;
          }
        }
        if (insertDiagonal)
        {
          // store the diagonal entry
          for (Plato::OrdinalType iDof=0; iDof<DofsPerNode; iDof++){
            columnIndices(numDofsSquared*R0+DofsPerNode*i+iDof) = DofsPerNode*rowNumber+iDof;
          }
          i_oh--; // i_oh lags i by 1 after we hit the diagonal
        }
        else
        {
          for (Plato::OrdinalType iDof=0; iDof<DofsPerNode; iDof++){
            columnIndices(dofColOffset+DofsPerNode*i+iDof) = DofsPerNode*tNodeOrds(i_oh + entryOffset_oh)+iDof;
          }
        }
      }
      for (Plato::OrdinalType iDof=0; iDof<numDofsThisRow; iDof++)
      {
        for (Plato::OrdinalType jDof=1; jDof<DofsPerNode; jDof++){
          columnIndices(dofColOffset+jDof*numDofsThisRow+iDof) = columnIndices(dofColOffset+iDof);
        }
      }
    });

    auto retMatrix = Teuchos::rcp(new MatrixType( rowMap, columnIndices, entries ));
    return retMatrix;
}


} // end namespace Plato

#endif
