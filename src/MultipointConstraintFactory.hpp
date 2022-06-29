/*
 * MultipointConstraintFactory.hpp
 *
 *  Created on: May 26, 2020
 */
#ifndef MULTIPOINT_CONSTRAINT_FACTORY_HPP
#define MULTIPOINT_CONSTRAINT_FACTORY_HPP

#include "Tet4.hpp"
#include "Tet10.hpp"
#include "SpatialModel.hpp"
#include "PlatoUtilities.hpp"
#include "MultipointConstraint.hpp"
#include "TieMultipointConstraint.hpp"
#include "PbcMultipointConstraint.hpp"
#include "PbcMultipointConstraint_def.hpp"

#ifdef PLATO_HEX_ELEMENTS
#include "Hex8.hpp"
#include "Hex27.hpp"
#include "Quad4.hpp"
#endif

namespace Plato
{

/******************************************************************************//**
 * \brief Factory for creating multipoint constraints.
 *
**********************************************************************************/
class MultipointConstraintFactory
{
public:
    /******************************************************************************//**
    * \brief Multipoint constraint factory constructor.
    * \param [in] aParamList input parameter list
    **********************************************************************************/
    MultipointConstraintFactory(Teuchos::ParameterList& aParamList) :
            mParamList(aParamList){}

    /******************************************************************************//**
    * \brief Create a multipoint constraint.
    * \return multipoint constraint
    **********************************************************************************/
    std::shared_ptr<Plato::MultipointConstraint>
    create(const Plato::SpatialModel & aSpatialModel, const std::string& aName)
    {
        const std::string tType = mParamList.get<std::string>("Type");

        if("Tie" == tType)
        {
            return makeMultipointConstraint<Plato::TieMultipointConstraint>(aSpatialModel, aName, mParamList); 
        }
        else if("PBC" == tType)
        {
            return makeMultipointConstraint<Plato::PbcMultipointConstraint>(aSpatialModel, aName, mParamList); 
        }
        return nullptr;
    }

private:
    Teuchos::ParameterList& mParamList; /*!< Input parameter list */

    template<template <typename> typename MPCType>
    inline
    std::shared_ptr<MultipointConstraint>
    makeMultipointConstraint(
      const Plato::SpatialModel    & aSpatialModel,
      const std::string            & aName,
      const Teuchos::ParameterList & aInput
    ) const
    {
        auto tElementType = aSpatialModel.Mesh->ElementType();
        if( Plato::tolower(tElementType) == "tet10" ||
            Plato::tolower(tElementType) == "tetra10" )
        {
            return std::make_shared<MPCType<Plato::Tet10>>(aSpatialModel, aName, mParamList);
        }
        else
        if( Plato::tolower(tElementType) == "tetra"  ||
            Plato::tolower(tElementType) == "tetra4" ||
            Plato::tolower(tElementType) == "tet4" )
        {
            return std::make_shared<MPCType<Plato::Tet4>>(aSpatialModel, aName, mParamList);
        }
        else
        if( Plato::tolower(tElementType) == "hex8" ||
            Plato::tolower(tElementType) == "hexa8" )
        {
#ifdef PLATO_HEX_ELEMENTS
            return std::make_shared<MPCType<Plato::Hex8>>(aSpatialModel, aName, mParamList);
#else
            ANALYZE_THROWERR("Not compiled with hex8 elements.");
#endif
        }
        else
        if( Plato::tolower(tElementType) == "hex27" ||
            Plato::tolower(tElementType) == "hexa27" )
        {
#ifdef PLATO_HEX_ELEMENTS
            return std::make_shared<MPCType<Plato::Hex27>>(aSpatialModel, aName, mParamList);
#else
            ANALYZE_THROWERR("Not compiled with hex27 elements.");
#endif
        }
        else
        if( Plato::tolower(tElementType) == "quad4" )
        {
#ifdef PLATO_HEX_ELEMENTS
            return std::make_shared<MPCType<Plato::Quad4>>(aSpatialModel, aName, mParamList);
#else
            ANALYZE_THROWERR("Not compiled with quad4 elements.");
#endif
        }
        else
        if( Plato::tolower(tElementType) == "tri3" )
        {
            return std::make_shared<MPCType<Plato::Tri3>>(aSpatialModel, aName, mParamList);
        }
        else
        {
            std::stringstream ss;
            ss << "Unknown mesh type: " << tElementType;
            ANALYZE_THROWERR(ss.str());
        }
    }
};
// class MultipointConstraintFactory

}
// namespace Plato

#endif
