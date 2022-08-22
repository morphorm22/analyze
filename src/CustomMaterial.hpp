#ifndef CUSTOMMATERIAL_HPP
#define CUSTOMMATERIAL_HPP

#include "PlatoTypes.hpp"

#include <Teuchos_ParameterList.hpp>

#include "ExpressionEvaluator.hpp"

#include <Kokkos_Parallel.hpp>

#include <Sacado.hpp>

#include <cstdarg>

#define DO_KOKKOS 1

namespace Plato
{

/******************************************************************************/
/*!
  \brief Class for custom material models
*/
class CustomMaterial
/******************************************************************************/
{
public:
    CustomMaterial(const Teuchos::ParameterList& aParamList) {}
    virtual ~CustomMaterial() {}

//protected:
    virtual Plato::Scalar GetCustomExpressionValue(
        const Teuchos::ParameterList& paramList,
        const std::string equationName ) const;

    virtual Plato::Scalar GetCustomExpressionValue(
        const Teuchos::ParameterList& paramList,
        const Plato::OrdinalType equationIndex = (Plato::OrdinalType) -1,
        const std::string equationName = std::string("Equation") ) const;

    void getTypedValue( const Plato::Scalar val,
                        const size_t ith, const size_t size,
                        Plato::Scalar &value ) const {
      value = val;
    };

    void getTypedValue( const Plato::Scalar val,
                        const size_t ith, const size_t size,
                        Sacado::Fad::DFad<Plato::Scalar> &value ) const {
      value = Sacado::Fad::DFad<Plato::Scalar>(size, ith, val);
    };

    template< typename T >
    void getTypedValue( const Plato::Scalar val,
                        const size_t ith, const size_t size,
                        T &value ) const {
      ANALYZE_THROWERR( "Unknown type value requested." );
    };

    KOKKOS_INLINE_FUNCTION
    void localPrintf( Plato::Scalar val ) const
    {
      printf( "%f ", val );
    }

    template< typename T >
    KOKKOS_INLINE_FUNCTION
    void localPrintf( T val ) const
    {
      printf( "%f %f ", val.val(), val.dx(0) );
    }

    // Template method which does the real work.
    template< typename TYPE >
    void
    GetCustomExpressionValue( const Teuchos::ParameterList& paramList,
                              const std::string equationStr,
                                    TYPE & result ) const
    {
        const size_t nThreads = 1;
        const size_t nValues  = 1;

        // Create an expression evaluator and pass the parameter list so
        // variable values can be retrived.
        ExpressionEvaluator< Kokkos::View< TYPE **, Plato::UVMSpace>,
                             Kokkos::View< TYPE **, Plato::UVMSpace>,
                             Kokkos::View< Plato::Scalar *, Plato::UVMSpace>,
                             Plato::Scalar > expEval;

        // Parse the equation. The expression tree is held internally.
        expEval.parse_expression( equationStr.c_str() );

        // For all of the variables found in the expression and get
        // their values from the parameter list.
        const std::vector< std::string > variableNames =
          expEval.get_variables();

        expEval.setup_storage( nThreads, nValues );

        for( size_t i=0; i<variableNames.size(); ++i)
        {
           Plato::Scalar val = paramList.get<Plato::Scalar>( variableNames[i] );

           expEval.set_variable( variableNames[i].c_str(), val );
        }

        // std::cout << "________________________________" << std::endl
        //           << "expression : " << equationStr << std::endl;

        // std::cout << "________________________________" << std::endl;
        // expEval.print_expression( std::cout );

        // If a valid equation, evaluate it,
        if( expEval.valid_expression( true ) )
        {
          // std::cout << "________________________________" << std::endl;
          // expEval.print_variables( std::cout );

          Kokkos::View<TYPE **, Plato::UVMSpace> results("results", nThreads, nValues);

#ifdef DO_KOKKOS
          // Device - GPU
          Kokkos::parallel_for(Kokkos::RangePolicy<>(0, nThreads),
                               KOKKOS_LAMBDA(const Plato::OrdinalType & tCellOrdinal)
                               {
                                 expEval.evaluate_expression( tCellOrdinal, results );
                               }, "Compute");

          // Wait for the GPU to finish so to get the data on to the CPU.
          Kokkos::fence();
#else
          // Host - CPU
          for( size_t tCellOrdinal=0; tCellOrdinal<nThreads; ++tCellOrdinal )
            expEval.evaluate_expression( tCellOrdinal, results );
#endif

          // std::cout << "________________________________" << std::endl;
          // for( size_t i=0; i<nValues; ++i )
          // {
          //   std::cout << "results = ";

          //   for( size_t j=0; j<nThreads; ++j )
          //     std::cout << results(j,i) << "  ";

          //   std::cout << std::endl;
          // }

          result = results(0,0);
        }

        // Clear the temporary storage used in the expression
        // otherwise there will be memory leaks.
        expEval.clear_storage();
    };
};
// class CustomMaterial

} // namespace Plato

#endif
