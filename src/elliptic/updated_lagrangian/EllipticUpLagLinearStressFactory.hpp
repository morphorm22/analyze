#ifndef PLATO_ELLIPTIC_UPDATED_LAGRANGIAN_LINEAR_STRESS_FACTORY_HPP
#define PLATO_ELLIPTIC_UPDATED_LAGRANGIAN_LINEAR_STRESS_FACTORY_HPP

#include "EllipticUpLagLinearStress.hpp"

#ifdef PLATO_EXPRESSION
  #include "EllipticUpLagLinearStressExpression.hpp"
#endif

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************//**
 * \brief Linear Stress Factory for creating linear stress models.
 *
 * \tparam EvaluationType - the evaluation type
 *
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class EllipticUpLagLinearStressFactory
{
public:
    /******************************************************************************//**
    * \brief linear stress factory constructor.
    **********************************************************************************/
    EllipticUpLagLinearStressFactory() {}

    /******************************************************************************//**
    * \brief Create a linear stress functor.
    * \param [in] aMaterialInfo - a material element stiffness matrix or
                                  a material model interface
    * \param [in] aParamList - input parameter list
    * \return Teuchos reference counter pointer to the linear stress functor
    **********************************************************************************/
    template<typename MaterialInfoType >
    Teuchos::RCP<Plato::Elliptic::UpdatedLagrangian::AbstractEllipticUpLagLinearStress<EvaluationType, SimplexPhysics> > create(
        const MaterialInfoType aMaterialInfo,
        const Teuchos::ParameterList& aParamList)
    {
      // Look for a linear stress block.
      if( aParamList.isSublist("Custom Elasticity Model") )
      {
#ifdef PLATO_EXPRESSION
          return Teuchos::rcp( new Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStressExpression<EvaluationType, SimplexPhysics>
                             (aMaterialInfo, aParamList) );
#else
          THROWERR("Plato Analyze was not built with expression support. "
                   "Rebuild with the cmake EXPRESSION option ON")
#endif
      }
      else
      {
        return Teuchos::rcp( new Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStress<EvaluationType, SimplexPhysics>
                             (aMaterialInfo) );
      }
    }
};
// class EllipticUpLagLinearStressFactory

}// namespace UpdatedLagrangian

}// namespace Elliptic

}// namespace Plato
#endif

#ifdef PLATOANALYZE_1D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStressFactory, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStressFactory, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStressFactory, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 3)
#endif
