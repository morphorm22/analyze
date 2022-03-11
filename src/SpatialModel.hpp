#pragma once

#include <Teuchos_ParameterList.hpp>

#include "PlatoMesh.hpp"
#include "PlatoMask.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Base class for spatial domain
 */
class SpatialDomain
/******************************************************************************/
{
public:
    Plato::Mesh Mesh;     /*!< mesh database */

private:
    std::string         mElementBlockName;  /*!< element block name */
    std::string         mMaterialModelName; /*!< material model name */
    std::string         mSpatialDomainName; /*!< element block name */
    bool                mIsFixedBlock;      /*!< flag for fixed block */

    Plato::OrdinalVector mTotalElemLids;   /*!< List of local elements ids in this domain */
    Plato::OrdinalVector mMaskedElemLids;  /*!< List of local elements ids after application of a masked operation */

public:
    /******************************************************************************//**
     * \fn getDomainName
     * \brief Return domain name.
     * \return domain name
     **********************************************************************************/
    decltype(mSpatialDomainName) 
    getDomainName() const
    {
        return mSpatialDomainName;
    }

    /******************************************************************************//**
     * \fn setDomainName
     * \brief Set patial domain name.
     * \param [in] aName domain model name
     **********************************************************************************/
    void setDomainName(const std::string & aName)
    {
        mSpatialDomainName = aName;
    }

    /******************************************************************************//**
     * \fn getMaterialName
     * \brief Return material model name.
     * \return material model name
     **********************************************************************************/
    decltype(mMaterialModelName) 
    getMaterialName() const
    {
        return mMaterialModelName;
    }

    /******************************************************************************//**
     * \fn setMaterialName
     * \brief Set material model name.
     * \param [in] aName material model name
     **********************************************************************************/
    void setMaterialName(const std::string & aName)
    {
        mMaterialModelName = aName;
    }

    /******************************************************************************//**
     * \fn getElementBlockName
     * \brief Return element block name.
     * \return element block name
     **********************************************************************************/
    decltype(mElementBlockName) 
    getElementBlockName() const
    {
        return mElementBlockName;
    }

    /******************************************************************************//**
     * \fn setElementBlockName
     * \brief Set element block name.
     * \param [in] aName element block name
     **********************************************************************************/
    void setElementBlockName
    (const std::string & aName)
    {
        mElementBlockName = aName;
    }

    /******************************************************************************//**
     * \fn isFixedBlock
     * \brief Return whether block has fixed control field.
     **********************************************************************************/
    decltype(mIsFixedBlock) 
    isFixedBlock() const
    {
        return mIsFixedBlock;
    }

    /******************************************************************************//**
     * \fn numCells
     * \brief Return the number of cells.
     * \return number of cells
     **********************************************************************************/
    Plato::OrdinalType 
    numCells() const
    {
        return mMaskedElemLids.extent(0);
    }

    /******************************************************************************//**
     * \fn Plato::OrdinalType numNodes
     * \brief Returns the number of nodes in the mesh.
     * \return number of nodes
     **********************************************************************************/
    Plato::OrdinalType
    numNodes() const
    {
        return Mesh->NumNodes();
    }

    /******************************************************************************//**
     * \brief get cell ordinal list
     * Note: A const reference is returned to prevent the ref count from being modified.  
    **********************************************************************************/
    const Plato::OrdinalVector &
    cellOrdinals() const
    {
        return mMaskedElemLids;
    }

    /******************************************************************************//**
     * \fn cellOrdinals
     * \brief Set cell ordinals for this element block.
     * \param [in] aName element block name
     **********************************************************************************/
    void cellOrdinals(const std::string & aName)
    {
        this->setMaskLocalElemIDs(aName);
    }

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh Default mesh
     * \param [in] aInputParams Spatial model definition
     **********************************************************************************/
    SpatialDomain
    (      Plato::Mesh   aMesh,
     const std::string & aName) :
        Mesh(aMesh),
        mSpatialDomainName(aName),
        mIsFixedBlock(false)
    {}

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh        Default mesh
     * \param [in] aInputParams Spatial model definition
     * \param [in] aName        Spatial model name
     **********************************************************************************/
    SpatialDomain
    (      Plato::Mesh              aMesh,
     const Teuchos::ParameterList & aInputParams,
     const std::string            & aName) :
        Mesh(aMesh),
        mSpatialDomainName(aName),
        mIsFixedBlock(false)
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Apply mask to this Domain
     *        This function removes elements that have a mask value of zero in \p aMask.
     *        Subsequent calls to numCells() and cellOrdinals() refer to the reduced list.
     *        Call removeMask() to remove the mask, or applyMask(...) to apply a different
     *        mask.
     * \param [in] aMask Plato Mask specifying active/inactive nodes and elements
    **********************************************************************************/
    template<Plato::OrdinalType mSpatialDim>
    void applyMask
    (std::shared_ptr<Plato::Mask<mSpatialDim>> aMask)
    {
        using OrdinalT = Plato::OrdinalType;

        auto tMask = aMask->cellMask();
        auto tTotalElemLids = mTotalElemLids;
        auto tNumEntries = tTotalElemLids.extent(0);

        // how many non-zeros in the mask?
        Plato::OrdinalType tSum(0);
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,tNumEntries), 
        LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
        {
            auto tElemOrdinal = tTotalElemLids(aOrdinal);
            aUpdate += tMask(tElemOrdinal); 
        }, tSum);
        Kokkos::resize(mMaskedElemLids, tSum);

        auto tMaskedElemLids = mMaskedElemLids;

        // create a list of elements with non-zero mask values
        OrdinalT tOffset(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumEntries),
        KOKKOS_LAMBDA (const OrdinalT& aOrdinal, OrdinalT& aUpdate, const bool& tIsFinal)
        {
            auto tElemOrdinal = tTotalElemLids(aOrdinal);
            const OrdinalT tVal = tMask(tElemOrdinal);
            if( tIsFinal && tVal ) { tMaskedElemLids(aUpdate) = tElemOrdinal; }
            aUpdate += tVal;
        }, tOffset);
    }
        
    /******************************************************************************//**
     * \brief Remove applied mask.
     *        This function resets the element list in this domain to the original definition.
     *        If no mask has been applied, this function has no effect.
    **********************************************************************************/
    void
    removeMask()
    {
        Kokkos::deep_copy(mMaskedElemLids, mTotalElemLids);
    }
    
    void setMaskLocalElemIDs
    (const std::string& aBlockName)
    {
        auto tElemLids = Mesh->GetLocalElementIDs(aBlockName);
        auto tNumElems = tElemLids.size();
        mTotalElemLids = Plato::OrdinalVector("element list", tNumElems);
        mMaskedElemLids = Plato::OrdinalVector("masked element list", tNumElems);

        auto tTotalElemLids = mTotalElemLids;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumElems), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tTotalElemLids(aCellOrdinal) = tElemLids[aCellOrdinal];
        }, "get element ids");
        Kokkos::deep_copy(mMaskedElemLids, mTotalElemLids);
    }

    void 
    initialize
    (const Teuchos::ParameterList & aInputParams)
    {
        if(aInputParams.isType<std::string>("Element Block"))
        {
            mElementBlockName = aInputParams.get<std::string>("Element Block");
            this->cellOrdinals(mElementBlockName);
        }
        else
        {
            ANALYZE_THROWERR("Parsing new Domain. Required keyword 'Element Block' not found");
        }

        if(aInputParams.isType<std::string>("Material Model"))
        {
            mMaterialModelName = aInputParams.get<std::string>("Material Model");
        }
        else
        {
            ANALYZE_THROWERR("Parsing new Domain. Required keyword 'Material Model' not found");
        }
        if(aInputParams.isType<bool>("Fixed Control"))
        {
            mIsFixedBlock = aInputParams.get<bool>("Fixed Control");
        }

        this->setMaskLocalElemIDs(mElementBlockName);
    }
};
// class SpatialDomain

/******************************************************************************/
/*!
 \brief Spatial models contain the mesh, meshsets, domains, etc that define
 a discretized geometry.
 */
class SpatialModel
/******************************************************************************/
{
public:
    Plato::Mesh Mesh;     /*!< mesh database */

    std::vector<Plato::SpatialDomain> Domains; /*!< list of spatial domains, i.e. element blocks */

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh     Default mesh
     **********************************************************************************/
    SpatialModel(Plato::Mesh aMesh) : Mesh(aMesh) {}

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh Default mesh
     * \param [in] aInputParams Spatial model definition
     **********************************************************************************/
    SpatialModel(
              Plato::Mesh              aMesh,
        const Teuchos::ParameterList & aInputParams
    ) :
        Mesh(aMesh)
    {
        if (aInputParams.isSublist("Spatial Model"))
        {
            auto tModelParams = aInputParams.sublist("Spatial Model");
            if (!tModelParams.isSublist("Domains"))
            {
                ANALYZE_THROWERR("Parsing 'Spatial Model' parameter list. Required 'Domains' parameter sublist not found");
            }

            auto tDomainsParams = tModelParams.sublist("Domains");
            for (auto tIndex = tDomainsParams.begin(); tIndex != tDomainsParams.end(); ++tIndex)
            {
                const auto &tEntry = tDomainsParams.entry(tIndex);
                const auto &tMyName = tDomainsParams.name(tIndex);

                if (!tEntry.isList())
                {
                    ANALYZE_THROWERR("Parameter in 'Domains' parameter sublist within 'Spatial Model' parameter list not valid.  Expect lists only.");
                }

                Teuchos::ParameterList &tDomainParams = tDomainsParams.sublist(tMyName);
                Domains.push_back( { aMesh, tDomainParams, tMyName });
            }
        }
        else
        {
            ANALYZE_THROWERR("Parsing 'Plato Problem'. Required 'Spatial Model' parameter list not found");
        }
    }

    template <Plato::OrdinalType mSpatialDim>
    void applyMask
    (std::shared_ptr<Plato::Mask<mSpatialDim>> aMask)
    {
        for( auto& tDomain : Domains )
        {
            tDomain.applyMask(aMask);
        }
    }

    /******************************************************************************//**
     * \brief Append spatial domain to spatial model.
     * \param [in] aDomain Spatial domain
     **********************************************************************************/
    void append
    (Plato::SpatialDomain & aDomain)
    {
        Domains.push_back(aDomain);
    }
};
// class SpatialModel

} // namespace Plato
