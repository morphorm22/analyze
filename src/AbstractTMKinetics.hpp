#ifndef PLATO_ABSTRACT_TMKINETICS_HPP
#define PLATO_ABSTRACT_TMKINETICS_HPP

#include "VoigtMap.hpp"
#include "FadTypes.hpp"
#include "MaterialModel.hpp"
#include "MaterialBasis.hpp"

namespace Plato
{

/******************************************************************************/
/*! Abstract Thermomechanics Kinetics functor.

    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<typename EvaluationType, typename ElementType>
class AbstractTMKinetics :
    public ElementType
{
protected:
    using StateT  = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
    using KinematicsScalarType = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    using ElementType::mNumSpatialDims;

    std::shared_ptr<Plato::UniformMaterialBasis<mNumSpatialDims>> mUniformMaterialBasis;
    std::shared_ptr<Plato::VaryingMaterialBasis<mNumSpatialDims>> mVaryingMaterialBasis;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model
    **********************************************************************************/
    AbstractTMKinetics(
        Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> const   aMaterialModel,
        Plato::SpatialDomain                                const & aSpatialDomain,
        Plato::DataMap                                      const & aDataMap
    )
    {
        UniformMaterialBasisFactory tUniformFactory;
        mUniformMaterialBasis = tUniformFactory.create(aMaterialModel, aSpatialDomain);

        VaryingMaterialBasisFactory tVaryingFactory;
        mVaryingMaterialBasis = tVaryingFactory.create<mNumSpatialDims>(aDataMap, aSpatialDomain);
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
        Plato::ScalarMultiVectorT<ControlScalarType> const & aControl) const
    {
        if(mUniformMaterialBasis)
        {
            mUniformMaterialBasis->VoigtTensorToMaterialBasis(aStrain);
            mUniformMaterialBasis->VectorToMaterialBasis(aTGrad);
        }
        if(mVaryingMaterialBasis)
        {
            mVaryingMaterialBasis->VoigtTensorToMaterialBasis(aStrain);
            mVaryingMaterialBasis->VectorToMaterialBasis(aTGrad);
        }

        this->compute(aStress, aFlux, aStrain, aTGrad, aTemperature, aControl);

        if(mVaryingMaterialBasis)
        {
            mVaryingMaterialBasis->VoigtTensorFromMaterialBasis(aStress);
            mVaryingMaterialBasis->VectorFromMaterialBasis(aFlux);
            // rotate the kinematics back for output
            mVaryingMaterialBasis->VoigtTensorFromMaterialBasis(aStrain);
            mVaryingMaterialBasis->VectorFromMaterialBasis(aTGrad);
        }
        if(mUniformMaterialBasis)
        {
            mUniformMaterialBasis->VoigtTensorFromMaterialBasis(aStress);
            mUniformMaterialBasis->VectorFromMaterialBasis(aFlux);
            // rotate the kinematics back for output
            mUniformMaterialBasis->VoigtTensorFromMaterialBasis(aStrain);
            mUniformMaterialBasis->VectorFromMaterialBasis(aTGrad);
        }
    };

    /***********************************************************************************
     * \brief Compute stress and thermal flux from strain, temperature, and temperature gradient
     * \param [in] aStrain infinitesimal strain tensor
     * \param [in] aTGrad temperature gradient
     * \param [in] aTemperature temperature
     * \param [out] aStress Cauchy stress tensor
     * \param [out] aFlux thermal flux vector
     **********************************************************************************/
    virtual void
    compute(
        Plato::ScalarArray3DT<KineticsScalarType>    const & aStress,
        Plato::ScalarArray3DT<KineticsScalarType>    const & aFlux,
        Plato::ScalarArray3DT<KinematicsScalarType>  const & aStrain,
        Plato::ScalarArray3DT<KinematicsScalarType>  const & aTGrad,
        Plato::ScalarMultiVectorT<StateT>            const & aTemperature,
        Plato::ScalarMultiVectorT<ControlScalarType> const & aControl) const = 0;

};
// class AbstractTMKinetics

}// namespace Plato
#endif
