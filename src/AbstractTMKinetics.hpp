#ifndef PLATO_ABSTRACT_TMKINETICS_HPP
#define PLATO_ABSTRACT_TMKINETICS_HPP

#include "ExpInstMacros.hpp"
#include "SimplexThermomechanics.hpp"
#include "VoigtMap.hpp"
#include "MaterialModel.hpp"
#include "SimplexFadTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Abstract Thermomechanics Kinetics functor.

    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class AbstractTMKinetics :
    public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>
{
protected:
    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using KineticsScalarType = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using KinematicsScalarType = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>; /*!<   strain variables automatic differentiation type */
    using ControlScalarType = typename EvaluationType::ControlScalarType;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model
    **********************************************************************************/
    AbstractTMKinetics(const Teuchos::RCP<Plato::MaterialModel<EvaluationType::SpatialDim>> aMaterialModel) 
    {
    }

    /***********************************************************************************
     * \brief Compute stress and thermal flux from strain, temperature, and temperature gradient
     * \param [in] aStrain infinitesimal strain tensor
     * \param [in] aTGrad temperature gradient
     * \param [in] aTemperature temperature
     * \param [out] aStress Cauchy stress tensor
     * \param [out] aFlux thermal flux vector
     **********************************************************************************/
    //template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    virtual void
    operator()( Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aStress,
                Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aFlux,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aStrain,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aTGrad,
                Kokkos::View<StateT*,       Plato::MemSpace> const& aTemperature,
                const Plato::ScalarMultiVectorT <ControlScalarType> & aControl) const = 0;

};
// class AbstractTMKinetics

}// namespace Plato
#endif

/*
#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::AbstractTMKinetics  , Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::AbstractTMKinetics  , Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::AbstractTMKinetics  , Plato::SimplexThermomechanics, 3)
#endif
*/
