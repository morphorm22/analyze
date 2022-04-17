/*
 * PbcMultipointConstraint.hpp
 *
 *  Created on: September 22, 2020
 */

#include "PbcMultipointConstraint.hpp"

/****************************************************************************/
template<typename ElementT>
Plato::PbcMultipointConstraint<ElementT>::
PbcMultipointConstraint(const Plato::SpatialModel & aSpatialModel,
                        const std::string & aName, 
                        Teuchos::ParameterList & aParam) :
                        Plato::MultipointConstraint(aName)
/****************************************************************************/
{
    // parse translation vector
    bool tIsVector = aParam.isType<Teuchos::Array<Plato::Scalar>>("Vector");

    Plato::Array<ElementT::mNumSpatialDims> tTranslation;
    if ( tIsVector )
    {
        auto tVector = aParam.get<Teuchos::Array<Plato::Scalar>>("Vector");
        for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
        {
            tTranslation(iDim) = tVector[iDim];
        }
    }
    else
    {
        std::ostringstream tMsg;
        tMsg << "TRANSLATION VECTOR FOR PBC MULTIPOINT CONSTRAINT NOT PARSED: CHECK INPUT PARAMETER KEYWORDS.";
        ANALYZE_THROWERR(tMsg.str())
    }

    // parse RHS value
    mValue = aParam.get<Plato::Scalar>("Value");

    // parse child node set
    std::string tChildNodeSet = aParam.get<std::string>("Child");
    auto tChildNodeLids = aSpatialModel.Mesh->GetNodeSetNodes(tChildNodeSet);
    auto tNumberChildNodes = tChildNodeLids.size();
    
    // Fill in child nodes
    Kokkos::resize(mChildNodes, tNumberChildNodes);

    this->updateNodesets(tNumberChildNodes, tChildNodeLids);
    
    // map child nodes
    Plato::ScalarMultiVector tChildNodeLocations       ("child node locations",        ElementT::mNumSpatialDims, tNumberChildNodes);
    Plato::ScalarMultiVector tMappedChildNodeLocations ("mapped child node locations", ElementT::mNumSpatialDims, tNumberChildNodes);

    this->mapChildVertexLocations(aSpatialModel.Mesh, tTranslation, tChildNodeLocations, tMappedChildNodeLocations);

    // get parent domain element data
    std::string tParentDomainName = aParam.get<std::string>("Parent");
    Plato::OrdinalVector tDomainCellMap;
    bool tFindName = 0;
    for(auto& tDomain : aSpatialModel.Domains)
    {
        auto tName = tDomain.getDomainName();
        if( tName == tParentDomainName )
            tDomainCellMap = tDomain.cellOrdinals();
            tFindName = 1;
    }
    if( tFindName == 0 )
    {
        std::ostringstream tMsg;
        tMsg << "PARENT DOMAIN FOR PBC MULTIPOINT CONSTRAINT NOT FOUND.";
        ANALYZE_THROWERR(tMsg.str())
    }
    
    // find elements that contain mapped child node locations (in specified domain)
    Plato::OrdinalVector tParentElements("mapped elements", tNumberChildNodes);
    Plato::Geometry::findParentElements<ElementT, Plato::Scalar>
      (aSpatialModel.Mesh, tDomainCellMap, tChildNodeLocations, tMappedChildNodeLocations, tParentElements);

    // get global IDs of unique parent nodes
    Plato::OrdinalVector tParentGlobalLocalMap;
    this->getUniqueParentNodes(aSpatialModel.Mesh, tParentElements, tParentGlobalLocalMap);
    
    // fill in mpc matrix values
    this->setMatrixValues(aSpatialModel.Mesh, tParentElements, tMappedChildNodeLocations, tParentGlobalLocalMap);
}

/****************************************************************************/
template<typename ElementT>
void Plato::PbcMultipointConstraint<ElementT>::
get(OrdinalVector & aMpcChildNodes,
    OrdinalVector & aMpcParentNodes,
    Plato::CrsMatrixType::RowMapVectorT & aMpcRowMap,
    Plato::CrsMatrixType::OrdinalVectorT & aMpcColumnIndices,
    Plato::CrsMatrixType::ScalarVectorT & aMpcEntries,
    ScalarVector & aMpcValues,
    OrdinalType aOffsetChild,
    OrdinalType aOffsetParent,
    OrdinalType aOffsetNnz)
/****************************************************************************/
{
    auto tValue = mValue;
    auto tNumberChildNodes = mChildNodes.size();
    auto tNumberParentNodes = mParentNodes.size();

    // fill in parent nodes
    auto tMpcParentNodes = aMpcParentNodes;
    auto tParentNodes = mParentNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberParentNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tMpcParentNodes(aOffsetParent+nodeOrdinal) = tParentNodes(nodeOrdinal); // parent node ID
    }, "parent nodes");

    // fill in chuld nodes and constraint info
    const auto& tMpcRowMap = mMpcMatrix->rowMap();
    const auto& tMpcColumnIndices = mMpcMatrix->columnIndices();
    const auto& tMpcEntries = mMpcMatrix->entries();
      
    auto tMpcChildNodes = aMpcChildNodes;
    auto tRowMap = aMpcRowMap;
    auto tColumnIndices = aMpcColumnIndices;
    auto tEntries = aMpcEntries;
    auto tValues = aMpcValues;

    auto tChildNodes = mChildNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tMpcChildNodes(aOffsetChild+nodeOrdinal) = tChildNodes(nodeOrdinal); // child node ID

        auto tRowStart = tMpcRowMap(nodeOrdinal);
        auto tRowEnd = tMpcRowMap(nodeOrdinal+1);
        for(Plato::OrdinalType entryOrdinal = tRowStart; entryOrdinal<tRowEnd; entryOrdinal++)
        {
            tColumnIndices(aOffsetNnz + entryOrdinal) = aOffsetParent + tMpcColumnIndices(entryOrdinal); // column indices
            tEntries(aOffsetNnz + entryOrdinal) = tMpcEntries(entryOrdinal); // entries (constraint coefficients)
        }

        tRowMap(aOffsetChild + nodeOrdinal) = aOffsetNnz + tMpcRowMap(nodeOrdinal); // row map
        tRowMap(aOffsetChild + nodeOrdinal + 1) = aOffsetNnz + tMpcRowMap(nodeOrdinal + 1); // row map

        tValues(aOffsetChild + nodeOrdinal) = tValue; // constraint RHS
        
    }, "child nodes, mpc matrix, and rhs values");

}

/****************************************************************************/
template<typename ElementT>
void Plato::PbcMultipointConstraint<ElementT>::
updateLengths(OrdinalType& lengthChild,
              OrdinalType& lengthParent,
              OrdinalType& lengthNnz)
/****************************************************************************/
{
    auto tNumberChildNodes = mChildNodes.size();
    auto tNumberParentNodes = mParentNodes.size();
    const auto& tMpcColMap = mMpcMatrix->columnIndices();
    auto tNumberNonzero = tMpcColMap.size();

    lengthChild += tNumberChildNodes;
    lengthParent += tNumberParentNodes;
    lengthNnz += tNumberNonzero;

}

/****************************************************************************/
template<typename ElementT>
void Plato::PbcMultipointConstraint<ElementT>::
updateNodesets(const OrdinalType& tNumberChildNodes,
               const Plato::OrdinalVectorT<const Plato::OrdinalType>& tChildNodeLids)
/****************************************************************************/
{
    auto tChildNodes = mChildNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tChildNodes(nodeOrdinal) = tChildNodeLids(nodeOrdinal); // child node ID
    }, "Child node IDs");
}

/****************************************************************************/
template<typename ElementT>
void Plato::PbcMultipointConstraint<ElementT>::
mapChildVertexLocations(
        Plato::Mesh                               aMesh,
  const Plato::Array<ElementT::mNumSpatialDims>   aTranslation,
        Plato::ScalarMultiVector                & aLocations,
        Plato::ScalarMultiVector                & aMappedLocations
)
/****************************************************************************/
{
    auto tCoords = aMesh->Coordinates();
    auto tNumberChildNodes = mChildNodes.size();

    auto tChildNodes = mChildNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        Plato::OrdinalType childNode = tChildNodes(nodeOrdinal);
        for(size_t iDim=0; iDim < ElementT::mNumSpatialDims; ++iDim)
        {
            aMappedLocations(iDim, nodeOrdinal) = tCoords[childNode*ElementT::mNumSpatialDims+iDim] + aTranslation(iDim);
        }
    }, "get verts and apply map");
}

/****************************************************************************/
template<typename ElementT>
void Plato::PbcMultipointConstraint<ElementT>::
getUniqueParentNodes(Plato::Mesh           aMesh,
                     OrdinalVector & aParentElements,
                     OrdinalVector & aParentGlobalLocalMap)
/****************************************************************************/
{
    auto tNVerts = aMesh->NumNodes();
    auto tNumberParentElements = aParentElements.size();
    auto tNVertsPerElem = ElementT::mNumNodesPerCell;
    auto tCells2Nodes = aMesh->Connectivity();

    // initialize array for storing parent element vertex ordinals
    Plato::OrdinalVector tNodeCounter("parent node counting", tNVerts);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), tNodeCounter);

    // check for missing parent elements or parent nodes
    Plato::OrdinalType tNumMissingParent(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumberParentElements),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& aElemOrdinal, Plato::OrdinalType & aUpdate)
    {
        if ( aParentElements(aElemOrdinal) == -2 ) 
        {  
            Kokkos::atomic_increment(&aUpdate);
        }
    }, tNumMissingParent);
    if ( tNumMissingParent > 0 )
    {
        std::ostringstream tMsg;
        tMsg << "NO PARENT ELEMENT COULD BE FOUND FOR AT LEAST ONE CHILD NODE IN PBC MULTIPOINT CONSTRAINT. \n";
        ANALYZE_THROWERR(tMsg.str())
    }

    // fill in parent element vertex ordinals
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberParentElements), LAMBDA_EXPRESSION(Plato::OrdinalType iElemOrdinal)
    {
        Plato::OrdinalType tElement = aParentElements(iElemOrdinal); 
        for(Plato::OrdinalType iVertOrdinal=0; iVertOrdinal < tNVertsPerElem; ++iVertOrdinal)
        {
            Plato::OrdinalType tVertIndex = tCells2Nodes[tElement*tNVertsPerElem + iVertOrdinal];
            tNodeCounter(tVertIndex) = 1;
        }
    }, "mark 1 for parent element vertices");

    // get number of unique parent nodes
    Plato::OrdinalType tSum(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,tNVerts),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
    {
        aUpdate += tNodeCounter(aOrdinal);
    }, tSum);
    Kokkos::resize(mParentNodes,tSum);

    // fill in unique parent nodes
    Plato::OrdinalType tOffset(0);
    auto tParentNodes = mParentNodes;
    Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,tNVerts),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const Plato::OrdinalType tVal = tNodeCounter(iOrdinal);
        if( tIsFinal && tVal ) 
        { 
            tParentNodes(aUpdate) = iOrdinal; 
        }
        aUpdate += tVal;
    }, tOffset);

    // create map from global node ID to local parent node ID
    Kokkos::resize(aParentGlobalLocalMap,tNVerts);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), aParentGlobalLocalMap);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tSum), LAMBDA_EXPRESSION(Plato::OrdinalType parentOrdinal)
    {
        Plato::OrdinalType tGlobalVertId = tParentNodes(parentOrdinal);
        aParentGlobalLocalMap(tGlobalVertId) = parentOrdinal;
    }, "map from global vertex ID to local parent node ID");

}

/****************************************************************************/
template<typename ElementT>
void Plato::PbcMultipointConstraint<ElementT>::
setMatrixValues(
    Plato::Mesh                aMesh,
    OrdinalVector            & aParentElements,
    Plato::ScalarMultiVector & aMappedLocations,
    OrdinalVector            & aParentGlobalLocalMap
)
/****************************************************************************/
{
    auto tCells2Nodes = aMesh->Connectivity();

    auto tNumChildNodes = mChildNodes.size();
    auto tNumParentNodes = mParentNodes.size();
    
    // build rowmap
    Plato::CrsMatrixType::RowMapVectorT tRowMap("row map", tNumChildNodes+1);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType iRowOrdinal)
    {
        tRowMap(iRowOrdinal) = ElementT::mNumNodesPerCell;
    }, "nonzeros");

    Plato::OrdinalType tNumEntries(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes+1),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const Plato::OrdinalType tVal = tRowMap(iOrdinal);
        if( tIsFinal )
        {
          tRowMap(iOrdinal) = aUpdate;
        }
        aUpdate += tVal;
    }, tNumEntries);
    
    // determine column map and entries
    Plato::CrsMatrixType::OrdinalVectorT tColMap("column indices", tNumEntries);
    Plato::CrsMatrixType::ScalarVectorT tEntries("matrix entries", tNumEntries);

    Plato::Geometry::GetBasis<ElementT, Plato::Scalar> tGetBasis(aMesh);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType iRowOrdinal)
    {

        Plato::Array<ElementT::mNumNodesPerCell, Plato::Scalar> tBasis(0.0);
        Plato::Array<ElementT::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);

        auto iElemOrdinal = aParentElements(iRowOrdinal);
        for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
        {
            tInPoint(iDim) = aMappedLocations(iDim, iRowOrdinal);
        }

        // basis function values
        tGetBasis(iElemOrdinal, tInPoint, tBasis);

        // fill in colmap and entries
        auto iEntryOrdinal = tRowMap(iRowOrdinal);

        for(Plato::OrdinalType iNode=0; iNode<ElementT::mNumNodesPerCell; iNode++)
        {
            tColMap(iEntryOrdinal+iNode) = aParentGlobalLocalMap(tCells2Nodes[iElemOrdinal*ElementT::mNumNodesPerCell+iNode]);
            tEntries(iEntryOrdinal+iNode) = tBasis[iNode];
        }

    }, "colmap and entries");

    // fill in mpc matrix
    mMpcMatrix = Teuchos::rcp( new Plato::CrsMatrixType(tRowMap, tColMap, tEntries, tNumChildNodes, tNumParentNodes, 1, 1) );
}

