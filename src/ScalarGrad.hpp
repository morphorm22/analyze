#pragma once

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! \brief Scalar gradient functor.
 *
 *  Given a gradient matrix and scalar field, compute the scalar gradient.
 *
 ******************************************************************************/
template<typename ElementType>
class ScalarGrad
{
public:
    /***********************************************************************************
     * \brief Compute scalar field gradient
     * \param [in] aCellOrdinal cell ordinal
     * \param [in/out] aOutput scalar field gradient workset
     * \param [in] aScalarField scalar field workset
     * \param [in] aGradient configuration gradient workset
     **********************************************************************************/
    template<typename OutputScalarType, typename StateScalarType, typename ConfigScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
      Plato::OrdinalType                                                   aCellOrdinal,
            Plato::Array<ElementType::mNumSpatialDims, OutputScalarType> & aOutput,
            Plato::ScalarMultiVectorT<StateScalarType>                     aScalarField,
      const Plato::Matrix<ElementType::mNumNodesPerCell,
                          ElementType::mNumSpatialDims,
                          ConfigScalarType>                              & aGradient
    ) const
    {
        // compute scalar gradient
        //
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < ElementType::mNumSpatialDims; tDimIndex++)
        {
            aOutput(tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerCell; tNodeIndex++)
            {
                aOutput(tDimIndex) += aScalarField(aCellOrdinal, tNodeIndex)
                        * aGradient(tNodeIndex, tDimIndex);
            }
        }
    }

# ifdef NOPE /* update or delete below */
    /***********************************************************************************
     * \brief Compute scalar field gradient
     * \param [in] aCellOrdinal cell ordinal
     * \param [in/out] aOutput scalar field gradient workset
     * \param [in] aScalarField scalar field workset
     * \param [in] aGradient configuration gradient workset
     **********************************************************************************/
    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(Plato::OrdinalType aCellOrdinal,
               Kokkos::View<ScalarType**, Plato::Layout, Plato::MemSpace> aOutput,
               Kokkos::View<ScalarType**, Plato::Layout, Plato::MemSpace> aScalarField,
               Plato::Array<SpaceDim>* aConfigGrad) const
    {
        // compute scalar gradient
        //
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aOutput(aCellOrdinal, tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                aOutput(aCellOrdinal, tDimIndex) += aScalarField(aCellOrdinal, tNodeIndex)
                        * aConfigGrad[tNodeIndex][tDimIndex];
            }
        }
    }

    /***********************************************************************************
     * \brief Compute scalar field gradient
     * \param [in] aCellOrdinal cell ordinal
     * \param [in/out] aOutput scalar field gradient workset
     * \param [in] aScalarField scalar field workset
     * \param [in] aGradient configuration gradient workset
     **********************************************************************************/
    template<typename ScalarGradType, typename ScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(Plato::OrdinalType aCellOrdinal,
               Plato::ScalarMultiVectorT<ScalarGradType> aOutput,
               Plato::ScalarMultiVectorT<ScalarType> aScalarField,
               Plato::ScalarArray3DT<GradientScalarType> aGradient) const
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aOutput(aCellOrdinal, tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                aOutput(aCellOrdinal, tDimIndex) += aScalarField(aCellOrdinal, tNodeIndex)
                        * aGradient(aCellOrdinal, tNodeIndex, tDimIndex);
            }
        }
    }

    /***********************************************************************************
     * \brief Compute scalar field gradient
     *
     * \tparam ScalarGradType     POD type for 2-D Kokkos::View
     * \tparam ScalarType         POD type for 2-D Kokkos::View
     * \tparam GradientScalarType POD type for 3-D Kokkos::View
     *
     * \param [in]     aCellOrdinal    cell ordinal
     * \param [in]     aNumDofsPerNode number of degree of freedom per node
     * \param [in]     aScalarOffset   scalar degree of freedom offset
     * \param [in]     aScalarField    scalar field workset
     * \param [in]     aConfigGradient configuration gradient workset
     * \param [in/out] aScalarGradient scalar field gradient workset
     *
     **********************************************************************************/
    template<typename ScalarGradType, typename ScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::OrdinalType & aNumDofsPerNode,
               const Plato::OrdinalType & aScalarOffset,
               const Plato::ScalarMultiVectorT<ScalarType> & aScalarField,
               const Plato::ScalarArray3DT<GradientScalarType> & aConfigGradient,
               const Plato::ScalarMultiVectorT<ScalarGradType> & aScalarGradient) const
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aScalarGradient(aCellOrdinal, tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * aNumDofsPerNode + aScalarOffset;
                aScalarGradient(aCellOrdinal, tDimIndex) += aScalarField(aCellOrdinal, tLocalOrdinal)
                        * aConfigGradient(aCellOrdinal, tNodeIndex, tDimIndex);
            }
        }
    }
#endif // NOPE
};
// class ScalarGrad

}
// namespace Plato
