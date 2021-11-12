#pragma once

namespace Plato
{ 
class FixedDomainDofs
{
private:
    const Plato::OrdinalType mNumDofsPerNode;
    const Plato::OrdinalType mNumNodesPerCell;
    const Plato::OrdinalType mNumNodes;
    const Omega_h::LOs mCellToNodeMap;
    std::vector<std::string> mFixedDomainNames; 

public:
    FixedDomainDofs(Omega_h::Mesh          & aMesh,
                    Teuchos::ParameterList & aFixedDomainParams,
                    Plato::OrdinalType     aNumDofsPerNode, 
                    Plato::OrdinalType     aNumNodesPerCell) :
                    mNumNodes(aMesh.nverts()), 
                    mCellToNodeMap(aMesh.ask_elem_verts()),
                    mNumDofsPerNode(aNumDofsPerNode),
                    mNumNodesPerCell(aNumNodesPerCell)

    {
        if (mNumDofsPerNode > 1)
            THROWERR("In FixedDomainDofs constructor : Number of DOFs per node for Helmoltz physics is greater than 1.")

        for (auto tIndex = aFixedDomainParams.begin(); tIndex != aFixedDomainParams.end(); ++tIndex)
        {
            mFixedDomainNames.push_back(aFixedDomainParams.name(tIndex));
        }
    }

    void operator()(const Plato::SpatialModel & aSpatialModel,
                    Plato::LocalOrdinalVector & aBcDofs)
    {
        Plato::LocalOrdinalVector tFixedBlockNodes("Nodes in fixed blocks", mNumNodes);
        Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), tFixedBlockNodes);

        for(const auto& tDomain : aSpatialModel.Domains)
        {
            auto tBlockName = tDomain.getElementBlockName();
            if (this->isFixedDomain(tBlockName))
                this->markBlockNodes(tDomain, tFixedBlockNodes);
        }

        auto tNumUniqueNodes = this->getNumberOfUniqueNodes(tFixedBlockNodes);
        Kokkos::resize(aBcDofs, tNumUniqueNodes);
        this->storeUniqueNodes(tFixedBlockNodes, aBcDofs);
    }

    bool isFixedDomain(const std::string & aBlockName) 
    {
        for (auto iNameOrdinal(0); iNameOrdinal < mFixedDomainNames.size(); iNameOrdinal++)
        {
            if(mFixedDomainNames[iNameOrdinal] == aBlockName)
                return true;
        }
        return false;
    }

    void markBlockNodes(const Plato::SpatialDomain & aDomain, 
                        Plato::LocalOrdinalVector aMarkedNodes)
    {
        auto tDomainCells = aDomain.cellOrdinals();
        auto tCellToNodeMap = mCellToNodeMap;
        auto tNumNodesPerCell = mNumNodesPerCell;
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tDomainCells.size()), LAMBDA_EXPRESSION(Plato::OrdinalType iElemOrdinal)
        {
            Plato::OrdinalType tElement = tDomainCells(iElemOrdinal); 
            for(Plato::OrdinalType iVertOrdinal=0; iVertOrdinal < tNumNodesPerCell; ++iVertOrdinal)
            {
                Plato::OrdinalType tVertIndex = tCellToNodeMap[tElement*tNumNodesPerCell + iVertOrdinal];
                aMarkedNodes(tVertIndex) = 1;
            }
        }, "nodes in domain element set");
    }

    Plato::OrdinalType getNumberOfUniqueNodes(const Plato::LocalOrdinalVector & aNodeVector)
    {
        Plato::OrdinalType tSum(0);
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,aNodeVector.size()),
        LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
        {
            aUpdate += aNodeVector(aOrdinal);
        }, tSum);
        return tSum;
    }
    void storeUniqueNodes(const Plato::LocalOrdinalVector & aMarkedNodes,
                          Plato::LocalOrdinalVector       & aBcDofs)
    {
        Plato::OrdinalType tOffset(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,aMarkedNodes.size()),
        KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
        {
            const Plato::OrdinalType tVal = aMarkedNodes(iOrdinal);
            if( tIsFinal && tVal ) 
                aBcDofs(aUpdate) = iOrdinal; 
            aUpdate += tVal;
        }, tOffset);
    }

};

} // namespace Plato
