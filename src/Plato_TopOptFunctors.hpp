/*
 * Plato_TopOptFunctors.hpp
 *
 *  Created on: Feb 12, 2019
 */

#include "PlatoStaticsTypes.hpp"

#pragma once

namespace Plato
{

/******************************************************************************//**
 * \brief Compute cell/element mass, /f$ \sum_{i=1}^{N} \[M\] \{z\} /f$, where
 * /f$ \[M\] /f$ is the mass matrix, /f$ \{z\} /f$ is the control vector and
 * /f$ N /f$ is the number of nodes.
 * \param [in] aCellOrdinal cell/element index
 * \param [in] aBasisValues 1D container of cell basis functions
 * \param [in] aCellControls 2D container of cell controls
 * \return cell/element penalized mass
 **********************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, typename ControlType>
KOKKOS_INLINE_FUNCTION ControlType
cell_mass(
    const Plato::OrdinalType                     & aCellOrdinal,
    const Plato::Array<NumNodesPerCell>          & aBasisValues,
    const Plato::ScalarMultiVectorT<ControlType> & aCellControls
)
{
    ControlType tCellMass = 0.0;
    for(Plato::OrdinalType tIndex_I = 0; tIndex_I < NumNodesPerCell; tIndex_I++)
    {
        ControlType tNodalMass = 0.0;
        for(Plato::OrdinalType tIndex_J = 0; tIndex_J < NumNodesPerCell; tIndex_J++)
        {
            tNodalMass += aBasisValues(tIndex_I) * aBasisValues(tIndex_J) * aCellControls(aCellOrdinal, tIndex_J);
        }
        tCellMass += tNodalMass;
    }
    return (tCellMass);
}

/******************************************************************************//**
 * \brief Compute average cell density
 * \param [in] aCellOrdinal cell/element index
 * \param [in] aNumControls number of controls
 * \param [in] aCellControls 2D container of cell controls
 * \return average density for this cell/element
 **********************************************************************************/
template<Plato::OrdinalType NumControls, typename ControlType>
KOKKOS_INLINE_FUNCTION ControlType
cell_density(const Plato::OrdinalType & aCellOrdinal,
             const Plato::ScalarMultiVectorT<ControlType> & aCellControls)
{
    ControlType tCellDensity = 0.0;
    for(Plato::OrdinalType tIndex = 0; tIndex < NumControls; tIndex++)
    {
        tCellDensity += aCellControls(aCellOrdinal, tIndex);
    }
    tCellDensity /= NumControls;
    return (tCellDensity);
}
// function cell_density

/******************************************************************************//**
 * \brief Compute average cell density
 * \param [in] aCellOrdinal cell/element index
 * \param [in] aCellControls 2D container of cell controls
 * \return average density for this cell/element
 **********************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, typename ControlType>
KOKKOS_INLINE_FUNCTION ControlType
cell_density(
    const Plato::OrdinalType                     & aCellOrdinal,
    const Plato::ScalarMultiVectorT<ControlType> & aCellControls,
    const Plato::Array<NumNodesPerCell>          & aBasisValues
)
{
    ControlType tCellDensity = 0.0;
    for(Plato::OrdinalType tIndex = 0; tIndex < NumNodesPerCell; tIndex++)
    {
        tCellDensity += aCellControls(aCellOrdinal, tIndex) * aBasisValues(tIndex);
    }
    return (tCellDensity);
}
// function cell_density

/***************************************************************************//**
 * \brief Apply penalty, i.e. density penalty, to 2-D view
 *
 * \tparam Length      number of data entries for a given cell
 * \tparam ControlType penalty, as a Scalar
 * \tparam ResultType  multi-vector, as a 3-D Kokkos::View
 *
 * \param [in]     aCellOrdinal cell ordinal, i.e. index
 * \param [in]     aPenalty     material penalty
 * \param [in/out] aOutput      physical quantity to be penalized
*******************************************************************************/
template<Plato::OrdinalType Length, typename ControlType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
apply_penalty(const Plato::OrdinalType aCellOrdinal, const ControlType & aPenalty, const Plato::ScalarMultiVectorT<ResultType> & aOutput)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutput(aCellOrdinal, tIndex) *= aPenalty;
    }
}
// function apply_penalty

} // namespace Plato
