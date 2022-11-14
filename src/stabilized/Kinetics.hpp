#pragma once

#include "LinearElasticMaterial.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
/*! Two-field Elasticity functor.

 given: strain, pressure gradient, fine scale displacement, pressure

 compute: deviatoric stress, volume flux, cell stabilization
 */
/******************************************************************************/
template<typename ElementType>
class Kinetics : public ElementType
{
private:
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness; /*!< material matrix with lame constants */
    Plato::Scalar mBulkModulus, mShearModulus; /*!< shear and bulk moduli */

    const Plato::Scalar mPressureScaling; /*!< pressure scaling term */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    Kinetics(const Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> aMaterialModel) :
            mCellStiffness(aMaterialModel->getStiffnessMatrix()),
            mBulkModulus(0.0),
            mShearModulus(0.0),
            mPressureScaling(aMaterialModel->getPressureScaling())
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
        {
            mBulkModulus += mCellStiffness(0, tDimIndex);
        }
        mBulkModulus /= mNumSpatialDims;

        Plato::OrdinalType tNumShear = mNumVoigtTerms - mNumSpatialDims;
        for(Plato::OrdinalType tShearIndex = 0; tShearIndex < tNumShear; tShearIndex++)
        {
            mShearModulus += mCellStiffness(tShearIndex + mNumSpatialDims, tShearIndex + mNumSpatialDims);
        }
        mShearModulus /= tNumShear;
    }

    /***********************************************************************************
     * \brief Compute deviatoric stress, volume flux, cell stabilization
     * \param [in] aVolume volume
     * \param [in] aProjectedPGrad projected pressure gradient on H^{1}(\Omega)
     * \param [in] aPressure pressure workset on L^2(\Omega)
     * \param [in] aStrain displacement strains workset on H^{1}(\Omega)
     * \param [in] aPressureGrad pressure gradient workset on L^2(\Omega)
     * \param [out] aDevStress deviatoric stress workset
     * \param [out] aVolumeFlux volume flux workset
     * \param [out] aCellStabilization stabilization term workset
     **********************************************************************************/
    template<typename KineticsScalarType,
             typename KinematicsScalarType,
             typename NodeStateScalarType,
             typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        VolumeScalarType                                    const & aVolume,
        Plato::Array<mNumSpatialDims, NodeStateScalarType>  const & aProjectedPGrad,
        Plato::Array<mNumVoigtTerms, KinematicsScalarType>  const & aStrain,
        Plato::Array<mNumSpatialDims, KinematicsScalarType> const & aPressureGrad,
        KineticsScalarType                                        & aPressure,
        Plato::Array<mNumVoigtTerms, KineticsScalarType>          & aDevStress,
        KineticsScalarType                                        & aVolumeFlux,
        Plato::Array<mNumSpatialDims, KineticsScalarType>         & aCellStabilization
    ) const
    {
        // compute deviatoric stress, i.e. \sigma - \frac{1}{3}trace(\sigma)
        //
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aDevStress(tVoigtIndex_I) = static_cast<Plato::Scalar>(0.0);
            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aDevStress(tVoigtIndex_I) += aStrain(tVoigtIndex_J)
                        * mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
        KineticsScalarType tTraceStress(0.0);
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
        {
            tTraceStress += aDevStress(tDimIndex);
        }
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
        {
            aDevStress(tDimIndex) -= tTraceStress / static_cast<Plato::Scalar>(3.0);
        }
        // compute volume strain, i.e. \div(u)
        //
        KinematicsScalarType tVolumetricStrain = 0.0;
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
        {
            tVolumetricStrain += aStrain(tDimIndex);
        }

        // compute volume difference
        //
        aPressure *= mPressureScaling;
        aVolumeFlux = mPressureScaling * (tVolumetricStrain - aPressure / mBulkModulus);

        // compute cell stabilization
        //
        KinematicsScalarType tTau = pow(aVolume, static_cast<Plato::Scalar>(2.0 / 3.0))
                / (static_cast<Plato::Scalar>(2.0) * mShearModulus);
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
        {
            aCellStabilization(tDimIndex) = mPressureScaling
                    * tTau * (mPressureScaling * aPressureGrad(tDimIndex) - aProjectedPGrad(tDimIndex));
        }
    }
};
// class Kinetics

} // namespace Stabilized
} // namespace Plato
