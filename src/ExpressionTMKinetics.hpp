#ifndef PLATO_EXPRESSION_TMKINETICS_HPP
#define PLATO_EXPRESSION_TMKINETICS_HPP

#include "AbstractTMKinetics.hpp"
#include "ExpressionEvaluator.hpp"

namespace Plato
{

/******************************************************************************/
/*! Expression Thermomechanics Kinetics functor.

    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class ExpressionTMKinetics :
    public Plato::AbstractTMKinetics<EvaluationType, SimplexPhysics>
{
protected:
    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using KineticsScalarType = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using KinematicsScalarType = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>; /*!<   strain variables automatic differentiation type */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model
    **********************************************************************************/
    ExpressionTMKinetics(const Teuchos::RCP<Plato::MaterialModel<EvaluationType::SpatialDim>> aMaterialModel) :
            AbstractTMKinetics<EvaluationType, SimplexPhysics>(aMaterialModel)
    {
    //    mModelType = aMaterialModel->type();
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
    void
    operator()( Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aStress,
                Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aFlux,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aStrain,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aTGrad,
                Kokkos::View<StateT*,       Plato::MemSpace> const& aTemperature) const override
    {
        const Plato::OrdinalType tNumCells = aStrain.extent(0);
        Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
        {
        }, "Cauchy stress");
    }

};
// class ExpressionTMKinetics

}// namespace Plato
#endif

/*
#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 3)
#endif
*/
