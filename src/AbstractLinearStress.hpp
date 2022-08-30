#ifndef PLATO_ABSTRACT_LINEAR_STRESS_HPP
#define PLATO_ABSTRACT_LINEAR_STRESS_HPP

#include "LinearElasticMaterial.hpp"
#include "FadTypes.hpp"

#include "PlatoMathTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType, typename ElementType >
class AbstractLinearStress : public ElementType
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>; /*!< strain variables automatic differentiation type */

    using ElementType::mNumVoigtTerms; /*!< number of stress/strain terms */

    const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness; /*!< material stiffness matrix */

    Plato::Array<mNumVoigtTerms> mReferenceStrain; /*!< reference strain tensor */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    AbstractLinearStress(const Plato::Matrix<mNumVoigtTerms,
                                             mNumVoigtTerms> aCellStiffness) :
      mCellStiffness(aCellStiffness)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < mNumVoigtTerms; tIndex++)
        {
            mReferenceStrain(tIndex) = 0.0;
        }
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] const aMaterialModel material model interface
    **********************************************************************************/
    AbstractLinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
      mCellStiffness  (aMaterialModel->getStiffnessMatrix()),
      mReferenceStrain(aMaterialModel->getReferenceStrain())
    {
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    virtual void
    operator()(Plato::ScalarMultiVectorT<ResultT> const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT> const& aSmallStrain) const = 0;

};
// class AbstractLinearStress

}// namespace Plato
#endif
