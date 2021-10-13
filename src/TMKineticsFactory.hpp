#ifndef PLATO_TMKINETICS_FACTORY_HPP
#define PLATO_TMKINETICS_FACTORY_HPP

#include <LinearTMKinetics.hpp>
#include <NonLinearTMKinetics.hpp>
#include <ExpressionTMKinetics.hpp>

namespace Plato
{
/******************************************************************************//**
 * \brief TMKinetics Factory for creating TMKinetics variants.
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class TMKineticsFactory
{
public:
    /******************************************************************************//**
    * \brief TMKinetics factory constructor.
    **********************************************************************************/
    TMKineticsFactory() {}

    /******************************************************************************//**
    * \brief Create a TMKinetics functor.
    * \param [in] aMaterialInfo - a material element stiffness matrix or
                                  a material model interface
    * \param [in] aParamList - input parameter list
    * \return Teuchos reference counter pointer to the linear stress functor
    **********************************************************************************/
    Teuchos::RCP<Plato::AbstractTMKinetics<EvaluationType, SimplexPhysics> > create(
        const Teuchos::RCP<Plato::MaterialModel<EvaluationType::SpatialDim>> aMaterialModel)
    {
        Plato::MaterialModelType tModelType = aMaterialModel->type();
        if (tModelType == Plato::MaterialModelType::Nonlinear)
        {
            return Teuchos::rcp( new Plato::NonLinearTMKinetics<EvaluationType,
                                                               SimplexPhysics>
                             (aMaterialModel) );
        } else
        if (tModelType == Plato::MaterialModelType::Linear)
        {
            return Teuchos::rcp( new Plato::LinearTMKinetics<EvaluationType,
                                                               SimplexPhysics>
                             (aMaterialModel) );
        } else
        if (tModelType == Plato::MaterialModelType::Expression)
        {
            return Teuchos::rcp( new Plato::ExpressionTMKinetics<EvaluationType,
                                                               SimplexPhysics>
                             (aMaterialModel) );
        }
    }
};
// class TMKineticsFactory

}// namespace Plato
#endif

/*
#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::TMKineticsFactory, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::TMKineticsFactory, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::TMKineticsFactory, Plato::SimplexThermomechanics, 3)
#endif
*/
