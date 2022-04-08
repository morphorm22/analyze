#pragma once

#include <cassert>

#include "ImplicitFunctors.hpp"
#include "AnalyzeMacros.hpp"
#include "Assembly.hpp"

#include "geometric/EvaluationTypes.hpp"


namespace Plato
{

namespace Geometric
{

/******************************************************************************/
/*! Base class for workset functionality.
*/
/******************************************************************************/
template<typename ElementType>
class WorksetBase : public ElementType
{
protected:
    Plato::OrdinalType mNumCells; /*!< local number of elements */
    Plato::OrdinalType mNumNodes; /*!< local number of nodes */

    using ElementType::mNumControl;          /*!< number of control vectors, i.e. materials */
    using ElementType::mNumNodesPerCell;     /*!< number of nodes per element */
    using ElementType::mNumSpatialDims;

    using ControlFad = typename Plato::Geometry::FadTypes<ElementType>::ControlFad;
    using ConfigFad  = typename Plato::Geometry::FadTypes<ElementType>::ConfigFad;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell; 

    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControl,     mNumNodesPerCell> mControlEntryOrdinal;
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumNodesPerCell> mConfigEntryOrdinal;

    Plato::NodeCoordinate<mNumSpatialDims, mNumNodesPerCell> mNodeCoordinate;

public:
    /******************************************************************************//**
     * \brief Return number of cells
     * \return number of cells
    **********************************************************************************/
    decltype(mNumCells) numCells() const
    {
        return (mNumCells);
    }

    /******************************************************************************//**
     * \brief Return number of nodes
     * \return number of nodes
    **********************************************************************************/
    decltype(mNumNodes) numNodes() const
    {
        return (mNumNodes);
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
    **********************************************************************************/
    WorksetBase(Plato::Mesh aMesh) :
            mNumCells(aMesh->NumElements()),
            mNumNodes(aMesh->NumNodes()),
            mControlEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControl, mNumNodesPerCell>(aMesh)),
            mConfigEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumNodesPerCell>(aMesh)),
            mNodeCoordinate(Plato::NodeCoordinate<mNumSpatialDims, mNumNodesPerCell>(aMesh))
    {
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (scalar type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <Plato::Scalar> & aControlWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
        Plato::workset_control_scalar_scalar<mNumNodesPerCell>(
              aDomain, mControlEntryOrdinal, aControl, aControlWS);
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (scalar type), as a 2-D Kokkos::View
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <Plato::Scalar> & aControlWS
    ) const
    {
        Plato::workset_control_scalar_scalar<mNumNodesPerCell>(
            mNumCells, mControlEntryOrdinal, aControl, aControlWS);
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <ControlFad>    & aFadControlWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
        Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlFad>(
              aDomain, mControlEntryOrdinal, aControl, aFadControlWS);
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <ControlFad>    & aFadControlWS
    ) const
    {
        Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlFad>(
            mNumCells, mControlEntryOrdinal, aControl, aFadControlWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aConfigWS configuration workset (scalar type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
              Plato::ScalarArray3DT <Plato::Scalar> & aConfigWS,
        const Plato::SpatialDomain                  & aDomain
    ) const
    {
      Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>(
              aDomain, mNodeCoordinate, aConfigWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aConfigWS configuration workset (scalar type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
        Plato::ScalarArray3DT <Plato::Scalar> & aConfigWS
    ) const
    {
      Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>(
          mNumCells, mNodeCoordinate, aConfigWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aReturnValue configuration workset (AD type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
              Plato::ScalarArray3DT <ConfigFad> & aFadConfigWS,
        const Plato::SpatialDomain              & aDomain
    ) const
    {
      Plato::workset_config_fad<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigFad>(
              aDomain, mNodeCoordinate, aFadConfigWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aReturnValue configuration workset (AD type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
        Plato::ScalarArray3DT <ConfigFad> & aFadConfigWS
    ) const
    {
      Plato::workset_config_fad<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigFad>(
          mNumCells, mNodeCoordinate, aFadConfigWS);
    }
}; // class WorksetBase

} // namespace Geometric

} // namespace Plato
