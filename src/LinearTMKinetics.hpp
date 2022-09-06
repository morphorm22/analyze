#ifndef PLATO_LINEAR_TMKINETICS_HPP
#define PLATO_LINEAR_TMKINETICS_HPP

#include "AbstractTMKinetics.hpp"

namespace Plato
{

/******************************************************************************/
/*! Linear Thermomechanics Kinetics functor.

    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<typename EvaluationType, typename ElementType>
class LinearTMKinetics :
    public Plato::AbstractTMKinetics<EvaluationType, ElementType>
{
protected:
    using StateT  = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
    using KinematicsScalarType = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumSpatialDims;
    
    Plato::Rank4VoigtConstant<mNumSpatialDims> mElasticStiffnessConstant;
    Plato::TensorConstant<mNumSpatialDims> mThermalExpansivityConstant;
    Plato::TensorConstant<mNumSpatialDims> mThermalConductivityConstant;

    Plato::Scalar mRefTemperature;
    const Plato::Scalar mScaling;
    const Plato::Scalar mScaling2;
    Plato::VoigtMap<mNumSpatialDims> cVoigtMap;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model
    **********************************************************************************/
    LinearTMKinetics(const Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> aMaterialModel) :
            AbstractTMKinetics<EvaluationType, ElementType>(aMaterialModel),
            mRefTemperature(aMaterialModel->getScalarConstant("Reference Temperature")),
            mScaling(aMaterialModel->getScalarConstant("Temperature Scaling")),
            mScaling2(mScaling*mScaling)
    {
        mElasticStiffnessConstant = aMaterialModel->getRank4VoigtConstant("Elastic Stiffness");
        mThermalExpansivityConstant = aMaterialModel->getTensorConstant("Thermal Expansivity");
        mThermalConductivityConstant = aMaterialModel->getTensorConstant("Thermal Conductivity");
    }

    /***********************************************************************************
     * \brief Compute stress and thermal flux from strain, temperature, and temperature gradient
     * \param [in] aStrain infinitesimal strain tensor
     * \param [in] aTGrad temperature gradient
     * \param [in] aTemperature temperature
     * \param [out] aStress Cauchy stress tensor
     * \param [out] aFlux thermal flux vector
     **********************************************************************************/
    void
    operator()(
        Plato::ScalarArray3DT<KineticsScalarType>    const & aStress,
        Plato::ScalarArray3DT<KineticsScalarType>    const & aFlux,
        Plato::ScalarArray3DT<KinematicsScalarType>  const & aStrain,
        Plato::ScalarArray3DT<KinematicsScalarType>  const & aTGrad,
        Plato::ScalarMultiVectorT<StateT>            const & aTemperature,
        Plato::ScalarMultiVectorT<ControlScalarType> const & aControl
    ) const override
    {
        auto tScaling = mScaling;
        auto tScaling2 = mScaling2;
        auto tRefTemperature = mRefTemperature;
        auto& tThermalExpansivityConstant = mThermalExpansivityConstant;
        auto& tThermalConductivityConstant = mThermalConductivityConstant;
        auto& tElasticStiffnessConstant = mElasticStiffnessConstant;
        auto& tVoigtMap = cVoigtMap;
        const Plato::OrdinalType tNumCells = aStrain.extent(0);

        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute element kinematics", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            StateT tTemperature = aTemperature(iCellOrdinal, iGpOrdinal);
            // compute thermal strain
            //
            StateT tstrain[mNumVoigtTerms] = {0};
            for( int iDim=0; iDim<EvaluationType::SpatialDim; iDim++ ){
                tstrain[iDim] = tScaling * tThermalExpansivityConstant(tVoigtMap.I[iDim], tVoigtMap.J[iDim])
                            * (tTemperature - tRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
                aStress(iCellOrdinal, iGpOrdinal, iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aStress(iCellOrdinal, iGpOrdinal, iVoigt) += (aStrain(iCellOrdinal, iGpOrdinal, jVoigt)-tstrain[jVoigt])*tElasticStiffnessConstant(iVoigt, jVoigt);
                }
            }

            // compute flux
            //
            for( int iDim=0; iDim<EvaluationType::SpatialDim; iDim++){
                aFlux(iCellOrdinal, iGpOrdinal, iDim) = 0.0;
                for( int jDim=0; jDim<EvaluationType::SpatialDim; jDim++){
                    aFlux(iCellOrdinal, iGpOrdinal, iDim) += tScaling2 * aTGrad(iCellOrdinal, iGpOrdinal, jDim)*tThermalConductivityConstant(iDim, jDim);
                }
            }
        });
    }
};
// class LinearTMKinetics

}// namespace Plato
#endif
