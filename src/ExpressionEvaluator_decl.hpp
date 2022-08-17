/*
  Original code provided by Rhyscitlema
  https://www.rhyscitlema.com/algorithms/expression-parsing-algorithm

  Modified to accept additional operations such log, pow, sqrt, abs
  etc. Also added the abilitiy to parse variables.
*/

#ifndef EXPRESSION_EVALUATOR_HPP
#define EXPRESSION_EVALUATOR_HPP

#include "AnalyzeMacros.hpp"
#include "PlatoTypes.hpp"
#include "ParseTools.hpp"

#include <Sacado.hpp>

#include <cmath>
#include <deque>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace Plato
{

// ************************************************************************* //
// Note: It is assumed that the ResultType, StateType, and VectorType
// are of type Kokkos::View so to properly handle the view of views
// de-referencing correctly. See clear_storage().
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
class ExpressionEvaluator
{
public:
  ExpressionEvaluator();
  ~ExpressionEvaluator();

  void initialize( Kokkos::View< VariableMap *, Plato::UVMSpace > & aVarMaps,
                   const Teuchos::ParameterList & aInputParams,
                   const std::vector< std::string > aParamLabels,
                   const Plato::OrdinalType nThreads,
                   const Plato::OrdinalType nValues );

  void    parse_expression( const char* expression );
  bool    valid_expression( const bool checkVariables = false ) const;
  KOKKOS_INLINE_FUNCTION
  void evaluate_expression( const Plato::OrdinalType thread,
                                  ResultType const & result ) const;
  void   delete_expression();
  void    print_expression(       std::ostream &os,
                            const bool print_val = false ) const;

  const std::vector< std::string > & get_variables() const;

  void     setup_storage( const Plato::OrdinalType nThreads,
                          const Plato::OrdinalType nValues );

  void     clear_storage() const;

  KOKKOS_INLINE_FUNCTION
  void     set_variable ( const char *, const ScalarType & value,
                          const Plato::OrdinalType thread = -1) const;
  KOKKOS_INLINE_FUNCTION
  void     set_variable ( const char *, const VectorType & values,
                          const Plato::OrdinalType thread = -1) const;
  KOKKOS_INLINE_FUNCTION
  void     set_variable ( const char *, const StateType  & values ) const;

  void   print_variables( std::ostream & os ) const;

  // Normally protect but using Kokkos so must be public.
  //protected:

// ************************************************************************* //
  enum struct NodeID  // Node arithmetic operation - note these are
                      // strongly typed (via the struct) because many
                      // of the enums are common words. Strongly
                      // typing them makes them unique.

                      // Note the underscore with SUBTRACTION_
                      // because of a conflict with the SUBTRACTION
                      // defined globally in EGADS: Electronic
                      // Geometry Aircraft Design System.
  {
    EMPTY_NODE,

    OPEN_PARENTHESIS,
    CLOSE_PARENTHESIS,

    OPEN_ABS_BAR,
    CLOSE_ABS_BAR,

    POSITIVE,
    NEGATIVE,

    ADDITION,
    SUBTRACTION_,
    MULTIPLICATION,
    DIVISION,

    EXPONENTIAL,
    LOG,
    POWER,
    SQRT,
    FACTORIAL,

    ABS,

    SIN,
    COS,
    TAN,

    NUMBER,
    VARIABLE
  };

// ************************************************************************* //
  enum struct NodeInfo  // Used for inserting nodes.
  {
    NoInfo,
    SkipClimbUp,
    RightAssociative,
    LeftAssociative
  };

// ************************************************************************* //
  enum DataSourceID  // Used for quick look up of variable data.
  {
    SCALAR_DATA_SOURCE = 0,
    VECTOR_DATA_SOURCE = 1,
     STATE_DATA_SOURCE = 2,
       MAX_DATA_SOURCE = 10  // Maximum number of variables in an expression
  };

  Plato::OrdinalType mNumDataSources{3};

// ************************************************************************* //
  // Map structure - used with Kokkos so hardwired.
  template< typename KEY_TYPE, typename VALUE_TYPE > struct _Map {
    KEY_TYPE key;
    VALUE_TYPE value;
  };

  template< typename KEY_TYPE, typename VALUE_TYPE >
  using Map = _Map< KEY_TYPE, VALUE_TYPE>;

// ************************************************************************* //
  // Node stucture for expression tree.
  typedef struct _Node {
    NodeID ID{ NodeID::EMPTY_NODE };     // Arithmetic operation

    Plato::OrdinalType precedence{ 0 };  // Precedence in the tree

    Plato::Scalar number{ 0 };           // Scalar value

    char variable[MAX_ARRAY_LENGTH];     // Variable name

    // Normally the tree would utilize pointers to its parent and
    // children nodes. But for execution on the GPU the nodes need to
    // be a defined chunk of memory. As such, instead of pointers, an
    // index is used instead.
    Plato::OrdinalType i_left  { (Plato::OrdinalType) -1 };
    Plato::OrdinalType i_right { (Plato::OrdinalType) -1 };
    Plato::OrdinalType i_parent{ (Plato::OrdinalType) -1 };

    // Index into the memory array for storing the intermediate
    // results.
    Plato::OrdinalType i_memory { (Plato::OrdinalType) -1 };

  } Node;

// ************************************************************************* //
  // All theses methods are support methods.
  void commute_expression();

  void traverse_expression();

  Plato::OrdinalType  insertNode(       Plato::OrdinalType i_current,
                                  const Plato::OrdinalType i_new,
                                  const NodeInfo info );

  NodeID validateNode( const Plato::OrdinalType i_node,
                       const bool checkVariables = false ) const;

  Plato::OrdinalType commuteNode( const Plato::OrdinalType i_node );

  void   traverseNode( const Plato::OrdinalType i_node,
                       const Plato::OrdinalType depth );

  KOKKOS_INLINE_FUNCTION
  bool   evaluateNode( const Plato::OrdinalType thread,
                       const Plato::OrdinalType i_node,
                             ResultType const & result ) const;

  void      clearNode( const Plato::OrdinalType i_node );
  void     deleteNode( const Plato::OrdinalType i_node );

  std::string printNodeID( const Plato::OrdinalType i_node,
                           const bool descriptor,
                           const bool print_val = false ) const;

  std::string printNode( const Plato::OrdinalType i_node,
                         const bool print_val = false ) const;

  void printNode(       std::ostream &os,
                  const Plato::OrdinalType i_node,
                  const int indent,
                  const bool print_val = false ) const;

  // ResultType factorial( ResultType n ) const;

// ************************************************************************* //
  // Member data

  // The index to the top level root node in the tree.
  Plato::OrdinalType mTreeRootNode{ (Plato::OrdinalType) -1 };

  // The total number of variable names in the equation.
  Plato::OrdinalType mNumVariables{ 0 };
  std::vector< std::string > mVariableList;

  // Number of threads to parallize over
  Plato::OrdinalType mNumThreads{ 0 };

  // Number of values to be evaluated.
  Plato::OrdinalType mNumValues{ 0 };

  // Total number of nodes used to construct the expression tree. Some
  // nodes may only be temporarily used. The array of nodes is
  // constructed on the host and used on the device.
  Plato::OrdinalType mNodesUsed{ 0 };
  Kokkos::View< Node *, Plato::UVMSpace > mNodes;

  // Total number of nodes in the expression tree and the post order
  // evaluation.  The array of nodes is constructed on the host and
  // used on the device.
  Plato::OrdinalType mNodeCount{ 0 };
  Kokkos::View< Plato::OrdinalType *, Plato::UVMSpace > mNodeOrder;

  // The maximum number of chunks of temporary memory needed.
  Plato::OrdinalType mNumMemoryChunks{ (Plato::OrdinalType) 0 };

  // A queue to hold indexes to the chunks of temporary memory.
  std::deque<Plato::OrdinalType> mMemQueue;

  // Array holding the results for the nodes. The space is reused
  // based on the on post order evaluation.
  Kokkos::View< ResultType *, Plato::UVMSpace > mResults;

  // A mapping of the variable names to their coresponding data in the
  // variable arrays - per thread, per variable.
  Kokkos::View< Map< char[MAX_ARRAY_LENGTH], Plato::OrdinalType > **,
                Plato::UVMSpace > mVariableMap;

  // Counts of the data stored in the variable arrays. Counts are need
  // because not all of the storage is used, per thread, per storage
  // (mNumDataSources).
  Kokkos::View< Plato::OrdinalType **, Plato::UVMSpace > mMapCounts;

  // Storage for variable data, there are three types, scalars are
  // constant and not indexed, vectors are indexed, and state values
  // are indexed by the thread and an index.
  Kokkos::View< ScalarType **, Plato::UVMSpace > mVariableScalarValues;
  Kokkos::View< VectorType **, Plato::UVMSpace > mVariableVectorValues;
  Kokkos::View< StateType   *, Plato::UVMSpace > mVariableStateValues;

  // Local definition for Kokkos
  KOKKOS_INLINE_FUNCTION int STRCMP (const char *p1, const char *p2) const
  {
    const unsigned char *s1 = (const unsigned char *) p1;
    const unsigned char *s2 = (const unsigned char *) p2;
    unsigned char c1, c2;

    do
    {
      c1 = (unsigned char) *s1++;
      c2 = (unsigned char) *s2++;
      if (c1 == '\0')
        return c1 - c2;
    }
    while (c1 == c2);

    return c1 - c2;
  }

  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( Plato::Scalar val ) const
  // {
  //   printf( "%f ", val );
  // }

  // template< typename T >
  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( T val ) const
  // {
  //   printf( "%f %f ", val.val(), val.dx(0) );
  // }

  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( const char * header, int line, int i, int j,
  //                   Plato::Scalar tVal, Plato::Scalar ttVal ) const
  // {
  //   printf( "scalar %s %d   %i %i   t=%f   tt=%f \n",
  //           header, line, i, j, tVal, ttVal );
  // }

  // template< typename T >
  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( const char * header, int line, int i, int j,
  //                   T tVal, Plato::Scalar ttVal ) const
  // {
  //   printf( "mixed %s %d  %i %i   t=%f %f   tt=%f %f \n",
  //           header, line, i, j, tVal.val(), tVal.dx(0), ttVal );
  // }

  // template< typename T >
  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( const char * header, int line,
  //                   int i, int j, T tVal, T ttVal ) const
  // {
  //   printf( "sacado %s %d  %i %i   t=%f %f   tt=%f %f \n",
  //           header, line, i, j, tVal.val(), tVal.dx(0), ttVal.val(), ttVal.dx(0) );
  // }
};



/******************************************************************************//**
 * \brief set_variable - Sets auxillary variables that are indexed.
                          and may change depending on the thread.
 * \param [in] varName - the variable name
 * \param [in] values  - the variable values
 * \param [in] thread  - the thread being executed
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
set_variable( const char * varName,
              const ScalarType & value,
              const Plato::OrdinalType thread ) const
{
  Plato::OrdinalType start, end;

  if( mNumThreads == 0 )
    GPU_WARNING( "Invalid call to set_variable - "
                 "setup_storage has not been called.",
                 "The number of threads has not been set." );

  // If the default set the value for all threads.
  if( thread == (Plato::OrdinalType) -1 )
  {
    start = 0;
    end = mNumThreads;
  }
  // Otherwise set just for this thread.
  else
  {
    start = thread;
    end = thread + 1;
  }

  for( Plato::OrdinalType t=start; t<end; ++t)
  {
    for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
    {
      if( STRCMP( mVariableMap(t, i).key, varName ) == 0 )
      {
        Plato::OrdinalType index = 0;

        if( mVariableMap(t, i).value != (Plato::OrdinalType) -1 )
        {
          // Value exists so replace it.
          index = mVariableMap(t, i).value % MAX_DATA_SOURCE;
        }
        else
        {
          // Value does not exists so add it.
          index = mMapCounts(t, SCALAR_DATA_SOURCE)++;

          // The index gives the index into the storage container. There
          // are three. The *_DATA_SOURCE * MAX_DATA_SOURCE gives the
          // index as to which of the three storage containers is being
          // used. The later makes for easy lookup when evaluating.
          mVariableMap(t, i).value = SCALAR_DATA_SOURCE * MAX_DATA_SOURCE + index;
        }

        mVariableScalarValues(t, index) = value;

        break;
      }
    }
  }
}

/******************************************************************************//**
 * \brief set_variable - Sets auxillary variables that are indexed.
                          and may change depending on the thread.
 * \param [in] varName - the variable name
 * \param [in] values  - the variable values
 * \param [in] thread  - the thread being executed
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
set_variable( const char * varName,
              const VectorType & values,
              const Plato::OrdinalType thread ) const
{
  Plato::OrdinalType start, end;

  if( mNumThreads == 0 )
    GPU_WARNING( "Invalid call to set_variable - "
                 "setup_storage has not been called.",
                 "The number of threads has not been set." );

  // If the default set the value for all threads.
  if( thread == (Plato::OrdinalType) -1 )
  {
    start = 0;
    end = mNumThreads;
  }
  // Otherwise set just for this thread.
  else
  {
    if( mNumValues == 1 )
    {
      GPU_WARNING( "Invalid call to set_variable - "
                   "The vector will be indexed over all threads but "
                   "requesting the vector be used on a per thread basis: ", varName);
    }

    start = thread;
    end = thread + 1;
  }

  // When the number of values is one the vector will be indexed
  // based on the thread.
  if( mNumValues == 1 && values.extent(0) != mNumThreads )
  {
    GPU_WARNING( "Invalid call to set_variable - "
                 "Indexing over the threads and the vector does not match: ", varName);
  }
  // When the number of values is greater than one the vector
  // will be indexed over the number values.
  else if( mNumValues > 1 && values.extent(0) != mNumValues )
  {
    GPU_WARNING( "Invalid call to set_variable - "
                 "Indexing over the number of values and the vector has the wrong length:", varName);
  }

  for( Plato::OrdinalType t=start; t<end; ++t)
  {
    for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
    {
      if( STRCMP( mVariableMap(t, i).key, varName ) == 0 )
      {
        Plato::OrdinalType index = 0;

        if( mVariableMap(t, i).value != (Plato::OrdinalType) -1 )
        {
          // Value exists so replace it.
          index = mVariableMap(t, i).value % MAX_DATA_SOURCE;
        }
        else
        {
          // Value does not exists so add it.
          index = mMapCounts(t, VECTOR_DATA_SOURCE)++;

          // The index gives the index into the storage container. There
          // are three. The *_DATA_SOURCE * MAX_DATA_SOURCE gives the
          // index as to which of the three storage containers is being
          // used. The later makes for easy lookup when evaluating.
          mVariableMap(t, i).value = VECTOR_DATA_SOURCE * MAX_DATA_SOURCE + index;
        }

        mVariableVectorValues(t, index) = values;

        break;
      }
    }
  }
}

/******************************************************************************//**
 * \brief set_variable - Sets input variables that are assumed to be multiple
                         vectors the indexing is across all threads.
 * \param [in] varName - the variable name
 * \param [in] values  - the variable values
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
set_variable( const char * varName,
              const StateType & values ) const
{
  if( mNumThreads == 0 )
      GPU_WARNING( "Invalid call to set_variable - "
                   "setup_storage has not been called.",
                   "The number of threads has not been set." );

  if( values.extent(0) != mNumThreads ||
      values.extent(1) != mNumValues )
  {
    GPU_WARNING( "Invalid call to set_variable - ", "");

    if( values.extent(0) != mNumThreads )
      GPU_WARNING( "The vector has the wrong length: ", varName);

    if( values.extent(1) != mNumValues )
      GPU_WARNING( "The vector has the wrong number of entries per thread: ", varName);

    if( mNumValues == 1 )
      GPU_WARNING( "When the number of values expected is one. "
                   "Set each value as a scalar constant "
                   "on a per thread basis.", "");
  }

  // Even though there is only a single input across all threads set
  // up the map for each thread so that the map can be used regardless
  // of which thread is being processed. This is opposed to an unique
  // input for each thread which is the case for a scalar and vector
  // values above.
  for( Plato::OrdinalType t = 0; t<mNumThreads; ++t)
  {
    for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
    {
      if( STRCMP( mVariableMap(t, i).key, varName ) == 0 )
      {
        Plato::OrdinalType index = 0;

        if( mVariableMap(t, i).value != (Plato::OrdinalType) -1 )
        {
          // Value exists so replace it.
          index = mVariableMap(t, i).value % MAX_DATA_SOURCE;
        }
        else
        {
          // Value does not exists so add it.
          index = mMapCounts(t, STATE_DATA_SOURCE)++;

          // The index gives the index into the storage container. There
          // are three. The *_DATA_SOURCE * MAX_DATA_SOURCE gives the
          // index as to which of the three storage containers is being
          // used. The later makes for easy lookup when evaluating.
          mVariableMap(t, i).value = STATE_DATA_SOURCE * MAX_DATA_SOURCE + index;
        }

        mVariableStateValues(index) = values;

        break;
      }
    }
  }
}

/******************************************************************************//**
 * \brief evaluate_expression - Evaluate the expression tree - public function.
 * \param [in]  thread - thread being evaluated.
 * \param [out] result - resulting data.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
evaluate_expression( const Plato::OrdinalType thread,
                           ResultType const & result ) const
{
#if USE_POST_ORDER
  // Post order evaluation of the nodes. Note the last node is the top
  // node which is evaluated separtately with the results going into
  // the return results instead of temporary results
  for( Plato::OrdinalType i=0; i<mNodeCount-1; ++i )
  {
    Plato::OrdinalType i_node = mNodeOrder[i];

    evaluateNode( thread, i_node, mResults[ mNodes[i_node].i_memory ] );
  }
#endif

  evaluateNode( thread, mTreeRootNode, result );
}

/******************************************************************************//**
 * \brief evaluateNode - Evaluate the current node - protected function.
 * \param [in] thread - thread being evaluated.
 * \param [in] i_node - index of the node.
 * \param [out] result - the expresion result.
 * \return evaluated - true if the node was evaluated.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
KOKKOS_INLINE_FUNCTION
bool
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
evaluateNode( const Plato::OrdinalType thread,
              const Plato::OrdinalType i_node,
                    ResultType const & result ) const
{
  // printf( "%d %d %d %d \n", __LINE__, thread, mNumValues, i_node );

  // At this point no error checks are needed as the tree has been
  // validated as part of the parsing of the expression.

  // This error should never happen, if it does happen it is a
  // developer error. A similar check is made in traverseNode which
  // throws an error. Thus commented out.
  // if( i_node == (Plato::OrdinalType) -1 )
  // {
  //   GPU_WARNING( "Invalid call to evaluateNode - "
  //                  "node index is -1", "" );

  //   return false;
  // }

  const Node & node = mNodes[i_node];

  // Empty node. This should never happen as checks are made not to
  // evaluate empty nodes. A similar check is made in traverseNode
  // which throws an error. Thus commented out.
  // if( node.ID == NodeID::EMPTY_NODE )
  // {
  //   GPU_WARNING( "Invalid call to evaluateNode - "
  //                  "node index is -1", itoa(i_ode) );

  //   return false;
  // }

  // Get the left side of the tree.
  ResultType left;
  if( node.i_left != (Plato::OrdinalType) -1 )
  {
    left = mResults[ mNodes[node.i_left].i_memory];

#if USE_RECURSION
    // Recursion - not used because it creates lots of warnings and
    // recursions blows the stack on the GPU.
    evaluateNode( thread, node.i_left, left );
#endif
  }

  // Get the right side of the tree.
  ResultType right;
  if( node.i_right != (Plato::OrdinalType) -1 )
  {
    right = mResults[ mNodes[node.i_right].i_memory ];

#if USE_RECURSION
    // Recursion - not used because it creates lots of
    // warnings and recursions blows the stack on the GPU.
    evaluateNode( thread, node.i_right, right );
#endif
  }

  // Divide by zero check.
  // if( node.ID == DIVISION )
   // {
  //   for( Plato::OrdinalType i = 0; i < mNumValues; ++ i )
  //   {
  //     if( right(thread,i) == 0 )
  //     {
  //       PRINTERR("Warning: Divide by zero.");
  //       result(thread,i) = 0;
  //     }
  //   }
  // }

  // Do the operation
  switch( node.ID )
  {
    case NodeID::POSITIVE:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = +right(thread,i);
      break;
    case NodeID::NEGATIVE:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = -right(thread,i);
      break;

    case NodeID::ADDITION:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = left(thread,i) + right(thread,i);
      break;
    case NodeID::SUBTRACTION_:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = left(thread,i) - right(thread,i);
      break;
    case NodeID::MULTIPLICATION:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = left(thread,i) * right(thread,i);
      break;
    case NodeID::DIVISION:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        if( right(thread,i) == 0 )
          result(thread,i) = 0;
        else
          result(thread,i) = left(thread,i) / right(thread,i);
      break;

    case NodeID::EXPONENTIAL:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = std::exp(right(thread,i));
      break;
    case NodeID::LOG:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = std::log(right(thread,i));
      break;
    case NodeID::POWER:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = std::pow(left(thread,i), right(thread,i));
      break;
    case NodeID::SQRT:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = std::sqrt(right(thread,i));
      break;
    case NodeID::ABS:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = std::abs(right(thread,i));
      break;
    // case NodeID::FACTORIAL:
      // for( Plato::OrdinalType i=0; i<mNumValues; ++i )
      //        result(thread,i) = factorial(left(thread,i));
      // break;
    case NodeID::SIN:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = std::sin(right(thread,i));
      break;
    case NodeID::COS:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = std::cos(right(thread,i));
      break;
    case NodeID::TAN:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = std::tan(right(thread,i));
      break;

    case NodeID::NUMBER:
    {
      const Plato::Scalar value = node.number;

      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = value;

      break;
    }

    case NodeID::VARIABLE:
    {
      Plato::OrdinalType type  = (Plato::OrdinalType) -1;
      Plato::OrdinalType index = (Plato::OrdinalType) -1;

      for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
      {
        if( STRCMP( mVariableMap(thread,i).key, node.variable ) == 0 )
        {
          if( mVariableMap(thread,i).value != (Plato::OrdinalType) -1 )
          {
            // Decode the value The type indicated the storage
            // container. There are three. The index gives the location
            // into the storage container being used.
            type  = mVariableMap(thread,i).value / MAX_DATA_SOURCE;
            index = mVariableMap(thread,i).value % MAX_DATA_SOURCE;
          }
          else
          {
            // std::stringstream errorMsg;
            // errorMsg << "Invalid call to evaluateNode - "
            //       << "can not find values for variable: " << node.variable;
            // ANALYZE_THROWERR( errorMsg.str() );

            GPU_WARNING( "Invalid call to evaluateNode - "
                         "can not find values for variable: ",
                         node.variable );
          }
        }
      }

      // Get the data from the storage container.
      if( type == SCALAR_DATA_SOURCE )
      {
        const ScalarType & value = mVariableScalarValues(thread, index);

        for( Plato::OrdinalType i=0; i<mNumValues; ++i )
          result(thread,i) = value;
      }
      else if( type == VECTOR_DATA_SOURCE )
      {
        const VectorType & values = mVariableVectorValues(thread, index);

        // When the number of values is one index based on the thread
        if( mNumValues == 1 )
        {
          result(thread,0) = values[thread];
        }
        // Otherwise index over all values.
        else
        {
          for( Plato::OrdinalType i=0; i<mNumValues; ++i )
            result(thread,i) = values[i];
        }
      }
      else if( type == STATE_DATA_SOURCE )
      {
        const StateType & values = mVariableStateValues(index);

        for( Plato::OrdinalType i=0; i<mNumValues; ++i )
          result(thread,i) = values(thread,i);
      }
      // This error should never happen, if it does it is a developer error.
      else
      {
        // std::stringstream errorMsg;
        // errorMsg << "Invalid call to evaluateNode - "
        //       << "can not find storage container for variable: " << variable;
        // ANALYZE_THROWERR( errorMsg.str() );

        GPU_WARNING( "Invalid call to evaluateNode - "
                     "can not find storage container for variable:",
                     node.variable );
      }

      break;
    }

    default:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = 0;
      break;
  }

  return true;
}

} // namespace Plato

#endif
