/*
 * TieMultipointConstraint.hpp
 *
 *  Created on: May 26, 2020
 */

/* #pragma once */

#ifndef TIE_MULTIPOINT_CONSTRAINT_HPP
#define TIE_MULTIPOINT_CONSTRAINT_HPP

#include <Teuchos_ParameterList.hpp>

#include "PlatoMesh.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "MultipointConstraint.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for tie multipoint constraint
 *
**********************************************************************************/
class TieMultipointConstraint : public Plato::MultipointConstraint
{
public:
    TieMultipointConstraint(const Plato::Mesh        aMesh,
                            const std::string      & aName, 
                            Teuchos::ParameterList & aParam);

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
             OrdinalType aOffsetNnz) override;
    
    // ! Get number of nodes in the constrained nodeset.
    void updateLengths(OrdinalType& lengthChild,
                       OrdinalType& lengthParent,
                       OrdinalType& lengthNnz) override;

    // ! Fill in node set members
    void updateNodesets(const OrdinalType& tNumberChildNodes,
                        const Plato::OrdinalVectorT<const Plato::OrdinalType>& tChildNodeLids,
                        const Plato::OrdinalVectorT<const Plato::OrdinalType>& tParentNodeLids);

private:
    OrdinalVector    mParentNodes;
    OrdinalVector    mChildNodes;
    Plato::Scalar         mValue;

};
// class TieMultipointConstraint

}
// namespace Plato

#endif
