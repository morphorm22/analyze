#ifndef BODYLOADS_HPP
#define BODYLOADS_HPP

#include <Teuchos_ParameterList.hpp>

#include "PlatoTypes.hpp"
#include "SpatialModel.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "PlatoMeshExpr.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Class for essential boundary conditions.
 */
template<typename EvaluationType, typename ElementType>
class BodyLoad
/******************************************************************************/
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumDofsPerNode = ElementType::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumNodesPerCell = ElementType::mNumNodesPerCell; /*!< number of nodes per cell/element */

protected:
    const std::string mName;
    const Plato::OrdinalType mDof;
    const std::string mFuncString;

public:

    /**************************************************************************/
    BodyLoad<EvaluationType, ElementType>(const std::string &aName, Teuchos::ParameterList &aParam) :
            mName(aName),
            mDof(aParam.get<Plato::OrdinalType>("Index", 0)),
            mFuncString(aParam.get<std::string>("Function"))
    {
    }
    /**************************************************************************/

    ~BodyLoad()
    {
    }

    /**************************************************************************/
    template<typename StateScalarType, typename ControlScalarType, typename ConfigScalarType, typename ResultScalarType>
    void
    get(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
        const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
              Plato::Scalar                                  aScale
    ) const
    {
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // map points to physical space
        //
        Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
        Plato::ScalarArray3DT<ConfigScalarType> tPhysicalPoints("cub points physical space", tNumCells, tNumPoints, mSpaceDim);

        Plato::mapPoints<ElementType>(aConfig, tPhysicalPoints);

        // get integrand values at quadrature points
        //
        Plato::ScalarMultiVectorT<ConfigScalarType> tFxnValues("function values", tNumCells*tNumPoints, 1);
        Plato::getFunctionValues<mSpaceDim>(tPhysicalPoints, mFuncString, tFxnValues);

        // integrate and assemble
        //
        auto tDof = mDof;
        Plato::VectorEntryOrdinal<mSpaceDim, mSpaceDim> tVectorEntryOrdinal(aSpatialDomain.Mesh);
        Kokkos::parallel_for("compute body load", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ResultScalarType tDensity(0.0);
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < ElementType::mNumNodesPerCell; tFieldOrdinal++)
            {
                tDensity += tBasisValues(tFieldOrdinal)*aControl(iCellOrdinal, tFieldOrdinal);
            }

            auto tEntryOffset = iCellOrdinal * tNumPoints;

            auto tFxnValue = tFxnValues(tEntryOffset + iGpOrdinal, 0);
            auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < ElementType::mNumNodesPerCell; tFieldOrdinal++)
            {
                Kokkos::atomic_add(&aResult(iCellOrdinal,tFieldOrdinal*mNumDofsPerNode+tDof),
                        tWeight * tFxnValue * tBasisValues(tFieldOrdinal) * tDensity);
            }
        });
    }

};
// end class BodyLoad

/******************************************************************************/
/*!
 \brief Owner class that contains a vector of BodyLoad objects.
 */
template<typename EvaluationType, typename ElementType>
class BodyLoads
/******************************************************************************/
{
private:
    std::vector<std::shared_ptr<BodyLoad<EvaluationType, ElementType>>> mBodyLoads;

public:

    /******************************************************************************//**
     * \brief Constructor that parses and creates a vector of BodyLoad objects based on
     *   the ParameterList.
     * \param aParams Body Loads sublist with input parameters
    **********************************************************************************/
    BodyLoads(Teuchos::ParameterList &aParams) :
            mBodyLoads()
    {
        for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
        {
            const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
            const std::string &tName = aParams.name(tIndex);

            if(!tEntry.isList())
            {
                ANALYZE_THROWERR("Parameter in Body Loads block not valid.  Expect lists only.");
            }

            Teuchos::ParameterList& tSublist = aParams.sublist(tName);
            std::shared_ptr<Plato::BodyLoad<EvaluationType, ElementType>> tBodyLoad;
            auto tNewBodyLoad = new Plato::BodyLoad<EvaluationType, ElementType>(tName, tSublist);
            tBodyLoad.reset(tNewBodyLoad);
            mBodyLoads.push_back(tBodyLoad);
        }
    }

    /**************************************************************************/
    /*!
     \brief Add the body load to the result workset
     */
    template<typename StateScalarType, typename ControlScalarType, typename ConfigScalarType, typename ResultScalarType>
    void
    get(
        const Plato::SpatialDomain                         & aSpatialDomain,
              Plato::ScalarMultiVectorT<StateScalarType>     aState,
              Plato::ScalarMultiVectorT<ControlScalarType>   aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT<ResultScalarType>    aResult,
              Plato::Scalar                                  aScale = 1.0
    ) const
    {
        for(const auto & tBodyLoad : mBodyLoads)
        {
            tBodyLoad->get(aSpatialDomain, aState, aControl, aConfig, aResult, aScale);
        }
    }
};

}

#endif
