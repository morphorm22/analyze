#pragma once

#include <sstream>

#include <Teuchos_ParameterList.hpp>

#include "PlatoMathExpr.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Class for essential boundary conditions.
 */
template<typename ElementType>
class EssentialBC
/******************************************************************************/
{
public:

    EssentialBC(
        const std::string            & aName,
              Teuchos::ParameterList & aParam,
              Plato::Scalar            aScaleFactor = 1.0
    ) :
        mName(aName),
        mNodeSetName(aParam.get <std::string>("Sides")),
        mDofIndex(aParam.get<Plato::OrdinalType>("Index", 0)),
        mMathExpr(nullptr),
        mScaleFactor(aScaleFactor)
    {
        this->checkValidInputs(aParam);
        this->initializeValueExpression(aParam);
    }

    void checkValidInputs(
        Teuchos::ParameterList & aParam)
    {
        if (aParam.isType<Plato::Scalar>("Value") && aParam.isType<std::string>("Function") )
        {
            ANALYZE_THROWERR("Specify either 'Value' or 'Function' in Boundary Condition definition");
        } 
        if (!aParam.isType<Plato::Scalar>("Value") && !aParam.isType<std::string>("Function") )
        {
            ANALYZE_THROWERR("Specify either 'Value' or 'Function' in Boundary Condition definition");
        } 
    }

    void initializeValueExpression(
        Teuchos::ParameterList & aParam)
    {
        if (aParam.isType<Plato::Scalar>("Value"))
        {
            mValue = aParam.get<Plato::Scalar>("Value");
        } 
        else if (aParam.isType<std::string>("Function"))
        {
            auto tValueExpr = aParam.get<std::string>("Function");
            mMathExpr = std::make_shared<Plato::MathExpr>(tValueExpr);
        }
    }

    /*!
     \brief Get the ordinals/values of the constrained nodeset.
     \param aMesh Plato mesh that contains the constrained nodeset.
     \param bcDofs Ordinal list to which the constrained dofs will be added.
     \param bcValues Value list to which the constrained value will be added.
     \param offset Starting location in bcDofs/bcValues where constrained dofs/values will be added.
     */
    void get(
        const Plato::Mesh          & aMesh,
              Plato::OrdinalVector & aBcDofs,
              Plato::ScalarVector  & aBcValues,
        const Plato::OrdinalType     aOffset,
        const Plato::Scalar          aTime=0.0)
    {
        auto tNodeIds = this->parseConstrainedNodeSets(aMesh);
        auto tValue = this->get_value(aTime);
        this->fillBcData(aBcDofs, aBcValues, tNodeIds, aOffset, tValue);
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    parseConstrainedNodeSets(
        const Plato::Mesh  aMesh)
    {
        auto tNodeLids = aMesh->GetNodeSetNodes(mNodeSetName);
        return tNodeLids;
    }

    Plato::Scalar get_value(
        const Plato::Scalar aTime=0.0) const
    {
        if (mMathExpr == nullptr)
        {
            return mValue / mScaleFactor;
        }
        else
        {
            return mMathExpr->value(aTime) / mScaleFactor;
        }
    }

    void fillBcData(
              Plato::OrdinalVector                            & aBcDofs,
              Plato::ScalarVector                             & aBcValues,
        const Plato::OrdinalVectorT<const Plato::OrdinalType> & aNodeIds,
        const Plato::OrdinalType                                aOffset,
        const Plato::Scalar                                     aValue)
    {
        auto tNumberConstrainedNodes = aNodeIds.size();
        constexpr Plato::OrdinalType tDofsPerNode = ElementType::mNumDofsPerNode;
        auto tDofIndex = mDofIndex;
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberConstrainedNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
        {
            aBcDofs(aOffset+aNodeOrdinal) = tDofsPerNode*aNodeIds[aNodeOrdinal]+tDofIndex;
            aBcValues(aOffset+aNodeOrdinal) = aValue;
        }, "Dirichlet BC");
    }

    Plato::OrdinalType get_length(
        const Plato::Mesh aMesh)
    {
        auto tNodeIds = this->parseConstrainedNodeSets(aMesh);
        auto tNumberConstrainedNodes = tNodeIds.size();
        this->checkForZeroLength(tNumberConstrainedNodes);

        return tNumberConstrainedNodes;
    }

    void checkForZeroLength(
        const Plato::Scalar aLength)
    {
        if (aLength == static_cast<Plato::OrdinalType>(0))
        {
            const std::string tErrorMessage = std::string("The set '") +
                mNodeSetName + "' specified in Essential Boundary Conditions contains 0 nodes.";
            WARNING(tErrorMessage)
        }
    }

    std::string const& get_ns_name() const
    {
        return mNodeSetName;
    }

private:
    const std::string mName;
    const std::string mNodeSetName;
    const Plato::OrdinalType mDofIndex;
    Plato::Scalar mValue;
    std::shared_ptr<Plato::MathExpr> mMathExpr;
    Plato::Scalar mScaleFactor;
};

} // namespace Plato

