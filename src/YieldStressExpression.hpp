#ifndef PLATO_YEILD_STRESS_EXPRESSION_HPP
#define PLATO_YEILD_STRESS_EXPRESSION_HPP

#include "AbstractYieldStress.hpp"

#include "ExpressionEvaluator.hpp"
#include "ParseTools.hpp"

/******************************************************************************/
/*
 To use the expression evaluator one must first add an expression to
 the analyzeInput.xml file as part of the 'Custom Plasticity Model':

      <ParameterList name='Custom Plasticity Model'>
        <Parameter  name='Equation' type='string' value='PenalizedHardeningModulusIsotropic * AccumulatedPlasticStrain + PenalizedInitialYieldStress'/>

        <Parameter  name='AccumulatedPlasticStrain' type='string' value='Local State Workset'/>

 Here the equation variable names, PenalizedHardeningModulusIsotropic
 and PenalizedInitialYieldStress are directly mapped to the parameter
 labels which are set as part of the Kokkos::view.

 Whereas the equation variable name, AccumulatedPlasticStrain is
 indirectly mapped to the parameter label for the 'Local State
 Workset' because the parameter label contains whitespace and
 variables names cannot.

 aLocalState must be passed in to the operator() regardless it if used
 or not. Whereas parameters are optional, currently zero to four may
 be passed in to the operator().

 Equation variables can also be fixed values:
        <Parameter  name='PHMI' type='double' value='0.01'/>

 The equation can also be from a Bingo file:

      <ParameterList name='Custom Plasticity Model'>
        <Parameter name="BingoFile" type="string" value="bingo.txt"/>

*/
/******************************************************************************/

namespace Plato
{
/******************************************************************************/
/*! Yield Stress Expression functor.
 *
 * \tparam EvaluationType - the evaluation type
v */
/******************************************************************************/
template<typename EvaluationType>
class YieldStressExpression :
    public Plato::AbstractYieldStress<EvaluationType>
{
protected:
    using LocalStateT = typename EvaluationType::LocalStateScalarType; /*!< local state variables automatic differentiation type */
    using ControlT    = typename EvaluationType::ControlScalarType;    /*!< control variables automatic differentiation type */
    using ResultT     = typename EvaluationType::ResultScalarType;     /*!< result variables automatic differentiation type */

    Teuchos::ParameterList mInputParams;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aInputParams Teuchos parameter list
    **********************************************************************************/
    YieldStressExpression(const Teuchos::ParameterList& aInputParams) :
      mInputParams(aInputParams)
    {
    }

    /******************************************************************************//**
     * \brief Compute the yield stress
     * \param [out] aResult - yield stress
     * \param [in]  aLocalState
     * \param [in]  aParameters
    **********************************************************************************/
    void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aResult,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Kokkos::View< Plato::ScalarVectorT< ControlT > *,
                             Plato::UVMSpace > const& aParameters) const override
  {
      // Method used with the factory and has it own Kokkos parallel_for
      const Plato::OrdinalType tNumCells = aResult.extent(0);
      const Plato::OrdinalType tNumTerms = aResult.extent(1);

      // Strings for mapping parameter names to the equation
      // variables. Note: the LocalState is a required parameter
      // though possibly not used. That is in the operator() just the
      // parameters after the aLocalState parameter are optional.
      std::vector< std::string > tParamLabels( aParameters.extent(0) );

      for( Plato::OrdinalType i=0; i<tParamLabels.size(); ++i )
      {
        tParamLabels[i] = aParameters(i).label();
      }

      // The local state label is always last.
      tParamLabels.push_back( aLocalState.label() );

      const Plato::OrdinalType tNumParamLabels = tParamLabels.size();

      // If the user wants to use the input parameters these hold the
      // names of the equation variables that are mapped to the input
      // parameters.
      Kokkos::View< VariableMap *, Plato::UVMSpace >
        tVarMaps ("Yield Stress Exp. Variable Maps", tNumParamLabels);

      /*!< expression evaluator */
      ExpressionEvaluator< Plato::ScalarMultiVectorT< ResultT >,
                           Plato::ScalarMultiVectorT< LocalStateT >,
                           Plato::ScalarVectorT< ControlT >,
                           ControlT > tExpEval;

      // Look for a Custom Plasticity Model
      if( mInputParams.isSublist("Custom Plasticity Model") )
      {
        auto tParams = mInputParams.sublist("Custom Plasticity Model");

        tExpEval.initialize(tVarMaps, tParams,
                            tParamLabels, tNumCells, tNumTerms );
      }
      // If for some reason the expression evalutor is called but
      // without the XML block.
      else
      {
        ANALYZE_THROWERR("Warning: Failed to find a 'Custom Plasticity Model' block.");
      }

      // The LocalState is a two-dimensional array with the first
      // indice being the cell index.
      if( tVarMaps(tNumParamLabels-1).key )
        tExpEval.set_variable( tVarMaps(tNumParamLabels-1).value, aLocalState );

      // Finally do the evaluation.
      Kokkos::parallel_for("Compute yield stress",
                           Kokkos::RangePolicy<>(0, tNumCells),
                           KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
      {
        // Values that change based on the aCellOrdinal index. These
        // are values that the user has requested to come from the
        // input parameters. The last is the LocalState and is handled
        // above, thus the reason for subtracting one.
        for( Plato::OrdinalType i=0; i<tNumParamLabels-1; ++i )
        {
          if( tVarMaps(i).key )
            tExpEval.set_variable( tVarMaps(i).value,
                                   (aParameters(i))(aCellOrdinal),
                                   aCellOrdinal );
        }

        // Evaluate the expression for this cell.
        tExpEval.evaluate_expression( aCellOrdinal, aResult );
      } );

      // Fence before deallocation on host, to make sure that the
      // device kernel is done first.
      Kokkos::fence();

      // Because there are views of views are used locally which are
      // reference counted and deleting the parent view DOES NOT
      // de-reference the child views a dummy view with no memory is
      // used to replace the child so it is de-referenced.  There is
      // still a slight memory leak because the creation of the dummy
      // views.
      Plato::ScalarVectorT< ControlT > tDummyVector
        ( "Yield Stress Exp. Dummy Parameter", 0 );

      // Drop all of the references to the parameter data.
      for( Plato::OrdinalType i=0; i<aParameters.extent(0); ++i )
      {
        aParameters(i) = tDummyVector;
      }

      // Clear the temporary storage used in the expression
      // otherwise there will be memory leaks.
      tExpEval.clear_storage();
  }
};
// class YieldStressExpression

}// namespace Plato
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexPlasticity,       2)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexPlasticity,       3)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexThermoPlasticity, 3)
#endif
