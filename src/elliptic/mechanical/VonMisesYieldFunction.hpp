/*
 * VonMisesYieldFunction.hpp
 *
 *  Created on: Feb 10, 2019
 */

#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Von Mises yield criterion class
**********************************************************************************/
template<Plato::OrdinalType NumSpatialDims, Plato::OrdinalType NumVoigtTerms>
class VonMisesYieldFunction
{
public:
    /******************************************************************************//**
     * \brief Constructor
    **********************************************************************************/
    VonMisesYieldFunction(){}

    /******************************************************************************//**
     * \brief Compute Von Mises yield criterion
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aGpOrdinal gauss point index
     * \param [in] aCauchyStress 2D container of cell Cauchy stresses
     * \param [out] aVonMisesStress 1D container of cell Von Mises yield stresses
    **********************************************************************************/
    template<typename InputType, typename ResultType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        const Plato::OrdinalType                   & aCellOrdinal,
        const Plato::OrdinalType                   & aGpOrdinal,
        const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
              ResultType                           & aVonMisesStress) const;

    template<typename InputType, typename ResultType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        const Plato::OrdinalType                   & aCellOrdinal,
        const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
              ResultType                           & aVonMisesStress) const;

    template<typename InputType, typename ResultType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        const Plato::OrdinalType                     & aCellOrdinal,
        const Plato::Array<NumVoigtTerms, InputType> & aCauchyStress,
              ResultType                             & aVonMisesStress) const;
};
// class VonMisesYieldFunction

/******************************************************************************//**
 * \brief Von Mises yield criterion for 3D problems
 *
 * \f$ sigma_{VM} = \sqrt{ \frac{ ( \sigma_{11} - sigma_{22} )^2 + ( \sigma_{22} - sigma_{33} )^2 +
 * ( \sigma_{33} - sigma_{11} )^2 + 6( sigma_{12}^2 + sigma_{23}^2 + sigma_{31}^2 ) }{2} }\f$
 *
 * \param [in] aCellOrdinal cell/element local ordinal
 * \param [in] aGpOrdinal gauss point local ordinal
 * \param [in] aCauchyStress cell/element Cauchy stress tensors
 * \param [out] aVonMisesStress cell/element Von Mises stresses
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<3,6>::operator()(
    const Plato::OrdinalType                   & aCellOrdinal,
    const Plato::OrdinalType                   & aGpOrdinal,
    const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
          ResultType                           & aVonMisesStress
) const
{
    ResultType tSigma11MinusSigma22 = aCauchyStress(aCellOrdinal, aGpOrdinal, 0) - aCauchyStress(aCellOrdinal, aGpOrdinal, 1);
    ResultType tSigma22MinusSigma33 = aCauchyStress(aCellOrdinal, aGpOrdinal, 1) - aCauchyStress(aCellOrdinal, aGpOrdinal, 2);
    ResultType tSigma33MinusSigma11 = aCauchyStress(aCellOrdinal, aGpOrdinal, 2) - aCauchyStress(aCellOrdinal, aGpOrdinal, 0);
    ResultType tSigma11MinusSigma22Squared = tSigma11MinusSigma22 * tSigma11MinusSigma22;
    ResultType tSigma22MinusSigma33Squared = tSigma22MinusSigma33 * tSigma22MinusSigma33;
    ResultType tSigma33MinusSigma11Squared = tSigma33MinusSigma11 * tSigma33MinusSigma11;
    ResultType tPrincipalStressContribution = tSigma11MinusSigma22Squared + tSigma22MinusSigma33Squared + tSigma33MinusSigma11Squared;

    ResultType tSigma33TimesSigma33 = aCauchyStress(aCellOrdinal, aGpOrdinal, 3) * aCauchyStress(aCellOrdinal, aGpOrdinal, 3);
    ResultType tSigma44TimesSigma44 = aCauchyStress(aCellOrdinal, aGpOrdinal, 4) * aCauchyStress(aCellOrdinal, aGpOrdinal, 4);
    ResultType tSigma55TimesSigma55 = aCauchyStress(aCellOrdinal, aGpOrdinal, 5) * aCauchyStress(aCellOrdinal, aGpOrdinal, 5);
    ResultType tShearStressContribution = static_cast<Plato::Scalar>(3)
            * (tSigma33TimesSigma33 + tSigma44TimesSigma44 + tSigma55TimesSigma55);

    ResultType tVonMises = (static_cast<Plato::Scalar>(0.5) * tPrincipalStressContribution) + tShearStressContribution;
    aVonMisesStress = sqrt(tVonMises);
}
template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<3,6>::operator()(
    const Plato::OrdinalType                   & aCellOrdinal,
    const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
          ResultType                           & aVonMisesStress
) const
{
    ResultType tSigma11MinusSigma22 = aCauchyStress(aCellOrdinal, 0) - aCauchyStress(aCellOrdinal, 1);
    ResultType tSigma22MinusSigma33 = aCauchyStress(aCellOrdinal, 1) - aCauchyStress(aCellOrdinal, 2);
    ResultType tSigma33MinusSigma11 = aCauchyStress(aCellOrdinal, 2) - aCauchyStress(aCellOrdinal, 0);
    ResultType tSigma11MinusSigma22Squared = tSigma11MinusSigma22 * tSigma11MinusSigma22;
    ResultType tSigma22MinusSigma33Squared = tSigma22MinusSigma33 * tSigma22MinusSigma33;
    ResultType tSigma33MinusSigma11Squared = tSigma33MinusSigma11 * tSigma33MinusSigma11;
    ResultType tPrincipalStressContribution = tSigma11MinusSigma22Squared + tSigma22MinusSigma33Squared + tSigma33MinusSigma11Squared;

    ResultType tSigma33TimesSigma33 = aCauchyStress(aCellOrdinal, 3) * aCauchyStress(aCellOrdinal, 3);
    ResultType tSigma44TimesSigma44 = aCauchyStress(aCellOrdinal, 4) * aCauchyStress(aCellOrdinal, 4);
    ResultType tSigma55TimesSigma55 = aCauchyStress(aCellOrdinal, 5) * aCauchyStress(aCellOrdinal, 5);
    ResultType tShearStressContribution = static_cast<Plato::Scalar>(3)
            * (tSigma33TimesSigma33 + tSigma44TimesSigma44 + tSigma55TimesSigma55);

    ResultType tVonMises = (static_cast<Plato::Scalar>(0.5) * tPrincipalStressContribution) + tShearStressContribution;
    aVonMisesStress = sqrt(tVonMises);
}
template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<3,6>::operator()(
    const Plato::OrdinalType         & aCellOrdinal,
    const Plato::Array<6, InputType> & aCauchyStress,
          ResultType                 & aVonMisesStress
) const
{
    ResultType tSigma11MinusSigma22 = aCauchyStress(0) - aCauchyStress(1);
    ResultType tSigma22MinusSigma33 = aCauchyStress(1) - aCauchyStress(2);
    ResultType tSigma33MinusSigma11 = aCauchyStress(2) - aCauchyStress(0);
    ResultType tSigma11MinusSigma22Squared = tSigma11MinusSigma22 * tSigma11MinusSigma22;
    ResultType tSigma22MinusSigma33Squared = tSigma22MinusSigma33 * tSigma22MinusSigma33;
    ResultType tSigma33MinusSigma11Squared = tSigma33MinusSigma11 * tSigma33MinusSigma11;
    ResultType tPrincipalStressContribution = tSigma11MinusSigma22Squared + tSigma22MinusSigma33Squared + tSigma33MinusSigma11Squared;

    ResultType tSigma33TimesSigma33 = aCauchyStress(3) * aCauchyStress(3);
    ResultType tSigma44TimesSigma44 = aCauchyStress(4) * aCauchyStress(4);
    ResultType tSigma55TimesSigma55 = aCauchyStress(5) * aCauchyStress(5);
    ResultType tShearStressContribution = static_cast<Plato::Scalar>(3)
            * (tSigma33TimesSigma33 + tSigma44TimesSigma44 + tSigma55TimesSigma55);

    ResultType tVonMises = (static_cast<Plato::Scalar>(0.5) * tPrincipalStressContribution) + tShearStressContribution;
    aVonMisesStress = sqrt(tVonMises);
}

/******************************************************************************//**
 * \brief Von Mises yield criterion for 2D problems (i.e. general plane stress)
 *
 * \f$ sigma_{VM} = \sqrt{ \sigma_{11}^2 - \sigma_{11}sigma_{22} + sigma_{22}^2 + 3sigma_{12}^2 } \f$
 *
 * \param [in] aCellOrdinal cell/element local ordinal
 * \param [in] aGpOrdinal gauss point local ordinal
 * \param [in] aCauchyStress cell/element Cauchy stress tensors
 * \param [out] aVonMisesStress cell/element Von Mises stresses
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<2,3>::operator()(
    const Plato::OrdinalType                   & aCellOrdinal,
    const Plato::OrdinalType                   & aGpOrdinal,
    const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
          ResultType                           & aVonMisesStress
) const
{
    ResultType tSigma11TimesSigma11 = aCauchyStress(aCellOrdinal, aGpOrdinal, 0) * aCauchyStress(aCellOrdinal, aGpOrdinal, 0);
    ResultType tSigma11TimesSigma22 = aCauchyStress(aCellOrdinal, aGpOrdinal, 0) * aCauchyStress(aCellOrdinal, aGpOrdinal, 1);
    ResultType tSigma22TimesSigma22 = aCauchyStress(aCellOrdinal, aGpOrdinal, 1) * aCauchyStress(aCellOrdinal, aGpOrdinal, 1);
    ResultType tSigma12TimesSigma12 = aCauchyStress(aCellOrdinal, aGpOrdinal, 2) * aCauchyStress(aCellOrdinal, aGpOrdinal, 2);

    ResultType tVonMises = tSigma11TimesSigma11 - tSigma11TimesSigma22 + tSigma22TimesSigma22
                           + (static_cast<Plato::Scalar>(3) * tSigma12TimesSigma12);
    aVonMisesStress = sqrt(tVonMises);
}

template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<2,3>::operator()(
    const Plato::OrdinalType                   & aCellOrdinal,
    const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
          ResultType                           & aVonMisesStress
) const
{
    ResultType tSigma11TimesSigma11 = aCauchyStress(aCellOrdinal, 0) * aCauchyStress(aCellOrdinal, 0);
    ResultType tSigma11TimesSigma22 = aCauchyStress(aCellOrdinal, 0) * aCauchyStress(aCellOrdinal, 1);
    ResultType tSigma22TimesSigma22 = aCauchyStress(aCellOrdinal, 1) * aCauchyStress(aCellOrdinal, 1);
    ResultType tSigma12TimesSigma12 = aCauchyStress(aCellOrdinal, 2) * aCauchyStress(aCellOrdinal, 2);

    ResultType tVonMises = tSigma11TimesSigma11 - tSigma11TimesSigma22 + tSigma22TimesSigma22
                           + (static_cast<Plato::Scalar>(3) * tSigma12TimesSigma12);
    aVonMisesStress = sqrt(tVonMises);
}

template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<2,3>::operator()(
    const Plato::OrdinalType         & aCellOrdinal,
    const Plato::Array<3, InputType> & aCauchyStress,
          ResultType                 & aVonMisesStress
) const
{
    ResultType tSigma11TimesSigma11 = aCauchyStress(0) * aCauchyStress(0);
    ResultType tSigma11TimesSigma22 = aCauchyStress(0) * aCauchyStress(1);
    ResultType tSigma22TimesSigma22 = aCauchyStress(1) * aCauchyStress(1);
    ResultType tSigma12TimesSigma12 = aCauchyStress(2) * aCauchyStress(2);

    ResultType tVonMises = tSigma11TimesSigma11 - tSigma11TimesSigma22 + tSigma22TimesSigma22
                           + (static_cast<Plato::Scalar>(3) * tSigma12TimesSigma12);
    aVonMisesStress = sqrt(tVonMises);
}

/******************************************************************************//**
 * \brief Von Mises yield criterion for 1D problems (i.e. uniaxial stress)
 *
 * \f$ sigma_{VM} = \sigma_{11} \f$
 *
 * \param [in] aCellOrdinal cell/element local ordinal
 * \param [in] aGpOrdinal gauss point local ordinal
 * \param [in] aCauchyStress cell/element Cauchy stress tensors
 * \param [out] aVonMisesStress cell/element Von Mises stresses
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<1,1>::operator()(
    const Plato::OrdinalType                   & aCellOrdinal,
    const Plato::OrdinalType                   & aGpOrdinal,
    const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
          ResultType                           & aVonMisesStress
) const
{
    aVonMisesStress = aCauchyStress(aCellOrdinal, aGpOrdinal, 0);
}
template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<1,1>::operator()(
    const Plato::OrdinalType                   & aCellOrdinal,
    const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
          ResultType                           & aVonMisesStress
) const
{
    aVonMisesStress = aCauchyStress(aCellOrdinal, 0);
}
template<>
template<typename InputType, typename ResultType>
KOKKOS_INLINE_FUNCTION void
VonMisesYieldFunction<1,1>::operator()(
    const Plato::OrdinalType         & aCellOrdinal,
    const Plato::Array<1, InputType> & aCauchyStress,
          ResultType                 & aVonMisesStress
) const
{
    aVonMisesStress = aCauchyStress(0);
}


} // namespace Plato
