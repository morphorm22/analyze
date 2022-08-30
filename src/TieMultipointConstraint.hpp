/*
 * TieMultipointConstraint.hpp
 *
 *  Created on: May 26, 2020
 */

/* #pragma once */

#ifndef TIE_MULTIPOINT_CONSTRAINT_HPP
#define TIE_MULTIPOINT_CONSTRAINT_HPP

#include <Teuchos_ParameterList.hpp>

#include "SpatialModel.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "MultipointConstraint.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for tie multipoint constraint
 *
**********************************************************************************/
template<typename ElementT>
class TieMultipointConstraint : public Plato::MultipointConstraint
{
public:
    TieMultipointConstraint(
      const Plato::SpatialModel & aSpatialModel,
      const std::string         & aName, 
            Teuchos::ParameterList    & aParam) :
      Plato::MultipointConstraint(aName)
/****************************************************************************/
{
    // parse RHS value
    mValue = aParam.get<Plato::Scalar>("Value");

    // parse child nodes
    std::string tChildNodeSet = aParam.get<std::string>("Child");
    auto tChildNodeLids = aSpatialModel.Mesh->GetNodeSetNodes(tChildNodeSet);
    auto tNumberChildNodes = tChildNodeLids.size();
    
    // parse parent nodes
    std::string tParentNodeSet = aParam.get<std::string>("Parent");
    auto tParentNodeLids = aSpatialModel.Mesh->GetNodeSetNodes(tParentNodeSet);
    auto tNumberParentNodes = tParentNodeLids.size();

    // Check that the number of child and parent nodes match
    if (tNumberChildNodes != tNumberParentNodes)
    {
        std::ostringstream tMsg;
        tMsg << "CHILD AND PARENT NODESETS FOR TIE CONSTRAINT NOT OF EQUAL LENGTH. \n";
        ANALYZE_THROWERR(tMsg.str())
    }

    // Fill in child and parent nodes
    Kokkos::resize(mChildNodes, tNumberChildNodes);
    Kokkos::resize(mParentNodes, tNumberParentNodes);

    this->updateNodesets(tNumberChildNodes, tChildNodeLids, tParentNodeLids);
}


    virtual ~TieMultipointConstraint(){}

    /*!
     \brief Get constraint matrix and RHS data.
     \param mpcRowMap CRS-style rowMap for constraint data.
     \param mpcColumnIndices CRS-style columnIndices for constraint data.
     \param mpcEntries CRS-style entries for constraint data.
     \param mpcValues Value list for constraint RHS.
     \param offsetChild Starting location in rowMap/RHS where constrained nodes/values will be added.
     \param offsetNnz Starting location in columnIndices/entries where constraining nodes/coefficients will be added.
     */
    void get(OrdinalVector & aMpcChildNodes,
             OrdinalVector & aMpcParentNodes,
             Plato::CrsMatrixType::RowMapVectorT & aMpcRowMap,
             Plato::CrsMatrixType::OrdinalVectorT & aMpcColumnIndices,
             Plato::CrsMatrixType::ScalarVectorT & aMpcEntries,
             ScalarVector & aMpcValues,
             OrdinalType aOffsetChild,
             OrdinalType aOffsetParent,
             OrdinalType aOffsetNnz) override
    {
        auto tValue = mValue;
        auto tNumberChildNodes = mChildNodes.size();

        // Fill in constraint info
        auto tMpcChildNodes = aMpcChildNodes;
        auto tMpcParentNodes = aMpcParentNodes;
        auto tRowMap = aMpcRowMap;
        auto tColumnIndices = aMpcColumnIndices;
        auto tEntries = aMpcEntries;
        auto tValues = aMpcValues;

        auto tChildNodes = mChildNodes;
        auto tParentNodes = mParentNodes;

        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
        {
            tMpcChildNodes(aOffsetChild + nodeOrdinal) = tChildNodes(nodeOrdinal); // child node ID
            tMpcParentNodes(aOffsetParent + nodeOrdinal) = tParentNodes(nodeOrdinal); // parent node ID

            tRowMap(aOffsetChild + nodeOrdinal) = aOffsetChild + nodeOrdinal; // row map
            tRowMap(aOffsetChild + nodeOrdinal + 1) = aOffsetChild + nodeOrdinal + 1; // row map

            tColumnIndices(aOffsetNnz + nodeOrdinal) = aOffsetParent + nodeOrdinal; // column indices (local parent node ID)
            tEntries(aOffsetNnz + nodeOrdinal) = 1.0; // entries (constraint coefficients)

            tValues(aOffsetChild + nodeOrdinal) = tValue; // constraint RHS

        }, "Tie constraint data");
    }
    
    // ! Get number of nodes in the constrained nodeset.
    void updateLengths(OrdinalType& lengthChild,
                       OrdinalType& lengthParent,
                       OrdinalType& lengthNnz) override
    {
        auto tNumberChildNodes = mChildNodes.size();
        auto tNumberParentNodes = mParentNodes.size();

        lengthChild += tNumberChildNodes;
        lengthParent += tNumberParentNodes;
        lengthNnz += tNumberChildNodes;
    }


    // ! Fill in node set members
    void updateNodesets(const OrdinalType& tNumberChildNodes,
                        const Plato::OrdinalVectorT<const Plato::OrdinalType>& tChildNodeLids,
                        const Plato::OrdinalVectorT<const Plato::OrdinalType>& tParentNodeLids)
    {
        auto tChildNodes = mChildNodes;
        auto tParentNodes = mParentNodes;

        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
        {
            tChildNodes(nodeOrdinal) = tChildNodeLids(nodeOrdinal); // child node ID
            tParentNodes(nodeOrdinal) = tParentNodeLids(nodeOrdinal); // parent node ID
        }, "Tie constraint data");
    }


private:
    OrdinalVector    mParentNodes;
    OrdinalVector    mChildNodes;
    Plato::Scalar         mValue;

};
// class TieMultipointConstraint

}
// namespace Plato

#endif
