/*
 * PbcMultipointConstraint.hpp
 *
 *  Created on: September 22, 2020
 */

/* #pragma once */

#ifndef PBC_MULTIPOINT_CONSTRAINT_HPP
#define PBC_MULTIPOINT_CONSTRAINT_HPP

#include <Teuchos_ParameterList.hpp>

#include "MultipointConstraint.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "BLAS1.hpp"
#include "Plato_MeshMapUtils.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for PBC multipoint constraint
 *
**********************************************************************************/
template<typename ElementT>
class PbcMultipointConstraint : public MultipointConstraint
{

public:
    PbcMultipointConstraint(const Plato::SpatialModel & aSpatialModel,
                            const std::string & aName, 
                            Teuchos::ParameterList & aParam);

    virtual ~PbcMultipointConstraint(){}

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
                        const Plato::OrdinalVectorT<const Plato::OrdinalType>& tChildNodeLids);

    // ! Perform translation mapping from child nodes to parent locations
    void
    mapChildVertexLocations(
            Plato::Mesh                               aMesh,
      const Plato::Array<ElementT::mNumSpatialDims>   aTranslation,
            Plato::ScalarMultiVector                & aLocations,
            Plato::ScalarMultiVector                & aMappedLocations);
    
    // ! Use mapped parent elements to find global IDs of unique parent nodes
    void getUniqueParentNodes(Plato::Mesh     aMesh,
                              OrdinalVector & aParentElements,
                              OrdinalVector & aParentGlobalLocalMap);

    void setMatrixValues(Plato::Mesh     aMesh,
                         OrdinalVector & aParentElements,
                         Plato::ScalarMultiVector & aMappedLocations,
                         OrdinalVector & aParentGlobalLocalMap);

private:
    OrdinalVector                      mChildNodes;
    OrdinalVector                      mParentNodes;
    Plato::Scalar                      mValue;
    Teuchos::RCP<Plato::CrsMatrixType> mMpcMatrix;

};
// class PbcMultipointConstraint

}
// namespace Plato

#endif
