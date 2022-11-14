#pragma once

#include "LinearElasticMaterial.hpp"
#include "PlatoMathTypes.hpp"
#include "AbstractLinearStress.hpp"

namespace Plato
{
/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType, typename ElementType >
class LinearStress :
    public Plato::AbstractLinearStress<EvaluationType, ElementType>
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>; /*!< strain variables automatic differentiation type */

    using ElementType::mNumVoigtTerms; /*!< number of stress/strain terms */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    LinearStress(const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> aCellStiffness) :
        AbstractLinearStress< EvaluationType, ElementType >(aCellStiffness)
    {
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    LinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
        AbstractLinearStress< EvaluationType, ElementType >(aMaterialModel)
    {
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    void
    operator()(Plato::ScalarMultiVectorT<ResultT> const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT> const& aSmallStrain) const override
  {
       // Method used to compute the stress with the factory and has
       // its own Kokkos parallel_for.

       // A lambda inside a member function captures the "this"
       // pointer not the actual members as such a local copy of the
       // data is need here for the lambda to capture everything.

       // If compiling with C++17 (Clang as the compiler or CUDA 11
       // with Kokkos 3.2). And using KOKKOS_CLASS_LAMBDA instead of
       // KOKKOS_EXPRESSION. Then the memeber data can be used
       // directly.
      const auto tCellStiffness   = this->mCellStiffness;
      const auto tReferenceStrain = this->mReferenceStrain;

      const Plato::OrdinalType tNumCells = aCauchyStress.extent(0);

      // Because the parallel_for loop is local, two dimensions of
      // parallelism can be exploited.
      Kokkos::parallel_for("Compute linear stress",
                           Kokkos::MDRangePolicy< Kokkos::Rank<2> >( {0, 0}, {tNumCells, mNumVoigtTerms} ),
                           KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal,
                                             const Plato::OrdinalType & tVoigtIndex_I)
      {
          aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

          for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
          {
              aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
                (aSmallStrain(aCellOrdinal, tVoigtIndex_J) -
                  tReferenceStrain(tVoigtIndex_J)) *
                tCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
          }
      } );
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<ResultT> const& aCauchyStress,
                                       Plato::ScalarMultiVectorT<StrainT> const& aSmallStrain) const
    {
        // Method used to compute the stress and called from within a
        // Kokkos parallel_for.
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
                  (aSmallStrain(aCellOrdinal, tVoigtIndex_J) -
                   this->mReferenceStrain(tVoigtIndex_J)) *
                  this->mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION void operator()(
              Plato::Array<mNumVoigtTerms, ResultT> & aCauchyStress,
        const Plato::Array<mNumVoigtTerms, StrainT> & aSmallStrain
    ) const
    {
        // Method used to compute the stress and called from within a
        // Kokkos parallel_for.
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(tVoigtIndex_I) +=
                  (aSmallStrain(tVoigtIndex_J) -
                   this->mReferenceStrain(tVoigtIndex_J)) *
                  this->mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }
};
// class LinearStress

}// namespace Plato
