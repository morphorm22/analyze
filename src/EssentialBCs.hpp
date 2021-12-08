#pragma once

#include <sstream>

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoMathExpr.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

#include "EssentialBC.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Owner class that contains a vector of EssentialBC objects.
 */
template<typename SimplexPhysicsType>
class EssentialBCs
/******************************************************************************/
{
public:

    /*!
     \brief Constructor that parses and creates a vector of EssentialBC objects
     based on the ParameterList.
     */
    EssentialBCs(
              Teuchos::ParameterList & aParams,
        const Omega_h::MeshSets      & aMeshSets
    ) : 
        mMeshSets(aMeshSets),
        mBCs()
    {
        for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
        {
            const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
            TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error, " Parameter in Boundary Conditions block not valid.  Expect lists only.");
        
            const std::string & tMyName = aParams.name(tIndex);
            Teuchos::ParameterList& tSublist = aParams.sublist(tMyName);
            this->constructEssentialBC(tSublist,tMyName);
        }
    }

    /*!
     \brief Constructor that parses and creates a vector of EssentialBC objects with scale factors
     */
    EssentialBCs(
              Teuchos::ParameterList                      & aParams,
        const Omega_h::MeshSets                           & aMeshSets, 
        const std::map<Plato::OrdinalType, Plato::Scalar> & aDofOffsetToScaleFactor
    ) :
        mMeshSets(aMeshSets),
        mBCs()
    {
        for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
        {
            const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
            TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error, " Parameter in Boundary Conditions block not valid.  Expect lists only.");

            const std::string & tMyName = aParams.name(tIndex);
            Teuchos::ParameterList& tSublist = aParams.sublist(tMyName);
            auto tScaleFactor = this->findScaleFactor(tSublist,aDofOffsetToScaleFactor);
            this->constructEssentialBC(tSublist,tMyName,tScaleFactor);
        }
    }

    void constructEssentialBC(
              Teuchos::ParameterList & aEssentialBCParams,
        const std::string            & aName,
              Plato::Scalar            aScaleFactor = 1.0)
    {
        const std::string tType = aEssentialBCParams.get<std::string>("Type");
        std::shared_ptr<EssentialBC<SimplexPhysicsType>> tMyBC;
        if("Zero Value" == tType)
        {
            const std::string tValueDocument = "solution component set to zero.";
            aEssentialBCParams.set("Value", static_cast<Plato::Scalar>(0.0), tValueDocument);
            tMyBC.reset(new EssentialBC<SimplexPhysicsType>(aName, aEssentialBCParams, aScaleFactor));
        }
        else if(tType == "Fixed Value" || tType == "Time Dependent")
            tMyBC.reset(new EssentialBC<SimplexPhysicsType>(aName, aEssentialBCParams, aScaleFactor));
        else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, " Boundary Condition type invalid: Not 'Zero Value' or 'Fixed Value'.");
        mBCs.push_back(tMyBC);
    }

    Plato::Scalar findScaleFactor(
              Teuchos::ParameterList                      & aEssentialBCParams,
        const std::map<Plato::OrdinalType, Plato::Scalar> & aDofOffsetToScaleFactor)
    {
        const Plato::OrdinalType tDofIndex = aEssentialBCParams.get<Plato::OrdinalType>("Index", 0);
        auto tSearch = aDofOffsetToScaleFactor.find(tDofIndex);
        if(tSearch != aDofOffsetToScaleFactor.end())
            return tSearch->second;
        return 1.0;
    }

    /*!
     \brief Get ordinals and values for constraints.
     \param [in/out] aBcDofs   Ordinals of all constrained dofs.
     \param [in/out] aBcValues Values of all constrained dofs.
     \param [in]     aTime     Current time (default=0.0).
     */
    void get(
        Plato::LocalOrdinalVector & aBcDofs,
        Plato::ScalarVector       & aBcValues,
        Plato::Scalar               aTime=0.0)
    {
        this->sizeBcArrays(aBcDofs,aBcValues);
        this->fillBcData(aBcDofs,aBcValues,aTime);
    }

    void sizeBcArrays(
        Plato::LocalOrdinalVector & aBcDofs,
        Plato::ScalarVector       & aBcValues)
    {
        Plato::OrdinalType tNumDofs(0);
        for(auto &tBC : mBCs)
        {
            tNumDofs += tBC->get_length(mMeshSets);
        }
        Kokkos::resize(aBcDofs, tNumDofs);
        Kokkos::resize(aBcValues, tNumDofs);
    }

    void fillBcData(
        Plato::LocalOrdinalVector & aBcDofs,
        Plato::ScalarVector       & aBcValues,
        Plato::Scalar               aTime)
    {
        Plato::OrdinalType tOffset(0);
        for(auto &tBC : mBCs)
        {
            tBC->get(mMeshSets, aBcDofs, aBcValues, tOffset, aTime);
            tOffset += tBC->get_length(mMeshSets);
        }
    }

    bool empty() const
    {
        return mBCs.empty();
    }

private:
    const Omega_h::MeshSets  & mMeshSets;
    std::vector<std::shared_ptr<EssentialBC<SimplexPhysicsType>>> mBCs;
};

} // namespace Plato

