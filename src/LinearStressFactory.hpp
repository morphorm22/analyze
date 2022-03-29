#ifndef PLATO_LINEAR_STRESS_FACTORY_HPP
#define PLATO_LINEAR_STRESS_FACTORY_HPP

#include "LinearStress.hpp"

#ifdef PLATO_EXPRESSION
  #include "LinearStressExpression.hpp"
#endif

namespace Plato
{
/******************************************************************************//**
 * \brief Linear Stress Factory for creating linear stress models.
 *
 * \tparam EvaluationType - the evaluation type
 *
**********************************************************************************/
template<typename EvaluationType, typename ElementType>
class LinearStressFactory
{
public:
    /******************************************************************************//**
    * \brief linear stress factory constructor.
    **********************************************************************************/
    LinearStressFactory() {}

    /******************************************************************************//**
    * \brief Create a linear stress functor.
    * \param [in] aMaterialInfo - a material element stiffness matrix or
                                  a material model interface
    * \param [in] aParamList - input parameter list
    * \return Teuchos reference counter pointer to the linear stress functor
    **********************************************************************************/
    template<typename MaterialInfoType >
    Teuchos::RCP<Plato::AbstractLinearStress<EvaluationType, ElementType> > create(
        const MaterialInfoType aMaterialInfo,
        const Teuchos::ParameterList& aParamList)
    {
      // Look for a linear stress block.
      if( aParamList.isSublist("Custom Elasticity Model") )
      {
#ifdef PLATO_EXPRESSION
        return Teuchos::rcp( new Plato::LinearStressExpression<EvaluationType, ElementType>
                             (aMaterialInfo, aParamList) );
#else
        ANALYZE_THROWERR("Plato Analyze was not built with expression support. "
                 "Rebuild with the cmake EXPRESSION option ON")
#endif
      }
      else
      {
        return Teuchos::rcp( new Plato::LinearStress<EvaluationType, ElementType>
                             (aMaterialInfo) );
      }
    }
};
// class LinearStressFactory

}// namespace Plato
#endif

#ifdef PLATOANALYZE_1D
//TODO PLATO_EXPL_DEC2(Plato::LinearStressFactory, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
//TODO PLATO_EXPL_DEC2(Plato::LinearStressFactory, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
//TODO PLATO_EXPL_DEC2(Plato::LinearStressFactory, Plato::SimplexMechanics, 3)
#endif
