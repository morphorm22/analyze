/*
 * InterpolateFromNodal.hpp
 *
 *  Created on: Feb 18, 2019
 */

#ifndef INTERPOLATE_FROM_NODAL_HPP_
#define INTERPOLATE_FROM_NODAL_HPP_

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***********************************************************************************
* 
* \brief InterpolateFromNodal Functor.
*
* Evaluate cell's nodal states at cubature points.
*
***********************************************************************************/
template<typename ElementType,
         Plato::OrdinalType NumDofsPerNode=ElementType::mNumSpatialDims,
         Plato::OrdinalType DofOffset=0,
         Plato::OrdinalType NumDofs=1>
class InterpolateFromNodal : public ElementType
{
    using ElementType::mNumNodesPerCell;

public:
    /*******************************************************************************
    * 
    * \brief Constructor
    *
    *******************************************************************************/
    InterpolateFromNodal()
    {
    }
    
    /*******************************************************************************
    * 
    * \brief Compute state values at cubature points 
    *
    * The state values are computed as follows: \hat{s} = \sum_{i=1}^{I}
    * \sum_\phi_{i} s_i, where \hat{s} is the state value, 
    * \phi_{i} is the array of basis functions.
    *
    * The input arguments are defined as:
    *
    *   \param aCellOrdinal      cell (i.e. element) ordinal 
    *   \param aBasisFunctions   cell interpolation functions
    *   \param aNodalCellStates  cell nodal states
    *   \param aStateValues      cell interpolated state at the cubature points 
    *
    *******************************************************************************/
    template<typename InStateType, typename OutStateType>
    DEVICE_TYPE inline void
    operator()(
        const Plato::OrdinalType                     & aCellOrdinal,
        const Plato::ScalarVector                    & aBasisFunctions,
        const Plato::ScalarMultiVectorT<InStateType> & aNodalCellStates,
        const Plato::ScalarVectorT<OutStateType>     & aStateValues) const
    {

        aStateValues(aCellOrdinal) = 0.0;

        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tCellDofIndex = (NumDofsPerNode * tNodeIndex) + DofOffset;
            aStateValues(aCellOrdinal) += aBasisFunctions(tNodeIndex) * aNodalCellStates(aCellOrdinal, tCellDofIndex);
        }
    }
    template<typename InStateType, typename OutStateType>
    DEVICE_TYPE inline void
    operator()(
        const Plato::OrdinalType                     & aCellOrdinal,
        const Plato::Array<mNumNodesPerCell>         & aBasisFunctions,
        const Plato::ScalarMultiVectorT<InStateType> & aNodalCellStates,
              OutStateType                           & aStateValue ) const
    {

        aStateValue = 0.0;

        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tCellDofIndex = (NumDofsPerNode * tNodeIndex) + DofOffset;
            aStateValue += aBasisFunctions(tNodeIndex) * aNodalCellStates(aCellOrdinal, tCellDofIndex);
        }
    }

    /*******************************************************************************
    *
    * \brief Compute state values at cubature points
    *
    * The state values are computed as follows: \hat{s} = \sum_{i=1}^{I}
    * \sum_\phi_{i} s_i, where \hat{s} is the state value,
    * \phi_{i} is the array of basis functions.
    *
    * The input arguments are defined as:
    *
    *   \param aCellOrdinal      cell (i.e. element) ordinal
    *   \param aBasisFunctions   cell interpolation functions
    *   \param aNodalCellStates  cell nodal states
    *   \param aStateValues      cell interpolated state at the cubature points
    *
    *******************************************************************************/
    template<typename InStateType, typename OutStateType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVector & aBasisFunctions,
                                       const Plato::ScalarMultiVectorT<InStateType> & aNodalCellStates,
                                       const Plato::ScalarMultiVectorT<OutStateType> & aStateValues) const
    {


        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofs; tDofIndex++)
        {
            aStateValues(aCellOrdinal, tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tCellDofIndex = (NumDofsPerNode * tNodeIndex) + DofOffset + tDofIndex;
                aStateValues(aCellOrdinal, tDofIndex) += aBasisFunctions(tNodeIndex) * aNodalCellStates(aCellOrdinal, tCellDofIndex);
            }
        }
    }
};
// class InterpolateFromNodal

} // namespace Plato

#endif
