#ifndef COMPUTED_FIELD_HPP
#define COMPUTED_FIELD_HPP

#include "PlatoMeshExpr.hpp"

namespace Plato
{

/******************************************************************************/
/*!
  \brief Class for computed fields.
*/
template<int SpaceDim, typename ScalarType=Plato::Scalar>
class ComputedField
/******************************************************************************/
{
  protected:
    const std::string   mName;
    const std::string   mFuncString;

    Plato::ScalarMultiVectorT<ScalarType> mValues;

  public:

  /**************************************************************************/
  ComputedField<SpaceDim, ScalarType>(
    const Plato::Mesh   aMesh,
    const std::string & aName,
    const std::string & aFunc) :
    mName(aName),
    mFuncString(aFunc),
    mValues(aName, aMesh->NumNodes(), /*num functions=*/1)
  /**************************************************************************/
  {
    initialize(aMesh, aName, aFunc);
  }

  /**************************************************************************/
  void initialize(
    const Plato::Mesh   aMesh,
    const std::string & aName,
    const std::string & aFunc)
  /**************************************************************************/
  {
    auto tNumPoints = aMesh->NumNodes();
    Plato::ScalarVectorT<ScalarType> tXcoords("x coordinates", tNumPoints);
    Plato::ScalarVectorT<ScalarType> tYcoords("y coordinates", tNumPoints);
    Plato::ScalarVectorT<ScalarType> tZcoords("z coordinates", tNumPoints);

    auto tCoords = aMesh->Coordinates();
    auto tValues = mValues;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumPoints), KOKKOS_LAMBDA(Plato::OrdinalType aPointOrdinal)
    {
      if (SpaceDim > 0) tXcoords(aPointOrdinal) = tCoords(aPointOrdinal*SpaceDim + 0);
      if (SpaceDim > 1) tYcoords(aPointOrdinal) = tCoords(aPointOrdinal*SpaceDim + 1);
      if (SpaceDim > 2) tZcoords(aPointOrdinal) = tCoords(aPointOrdinal*SpaceDim + 2);
    },"fill coords");

    ExpressionEvaluator<Plato::ScalarMultiVectorT<ScalarType>,
                        Plato::ScalarMultiVectorT<ScalarType>,
                        Plato::ScalarVectorT<ScalarType>,
                        Plato::Scalar> tExpEval;

    tExpEval.parse_expression(aFunc.c_str());
    tExpEval.setup_storage(tNumPoints, /*num vals to eval =*/ 1);

    // The coords are indexed by threads so set the values outside the
    // parallel for loop.
//    tExpEval.set_variable("x", tXcoords);
//    tExpEval.set_variable("y", tYcoords);
//    tExpEval.set_variable("z", tZcoords);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumPoints), KOKKOS_LAMBDA(Plato::OrdinalType aPointOrdinal)
    {
        // Set the coords as a constant on a per thread basis. This
        // call works but is not needed as the coords are indexed by
        // threads so set the values outside the parallel for loop.

        tExpEval.set_variable("x", tXcoords(aPointOrdinal), aPointOrdinal);
        tExpEval.set_variable("y", tYcoords(aPointOrdinal), aPointOrdinal);
        tExpEval.set_variable("z", tZcoords(aPointOrdinal), aPointOrdinal);

        // This call works but is not needed as values are used across
        // all threads so set the values outside the parallel for loop.

        // tExpEval.set_variable("x", tXcoords, aPointOrdinal);
        // tExpEval.set_variable("y", tYcoords, aPointOrdinal);
        // tExpEval.set_variable("z", tZcoords, aPointOrdinal);

        tExpEval.evaluate_expression( aPointOrdinal, tValues );
    }, "evaluate");
    Kokkos::fence();
    tExpEval.clear_storage();
  }

  ~ComputedField(){}

//  /******************************************************************************/
//  void get(Plato::ScalarMultiVectorT<ScalarValue> aValues)
//  /******************************************************************************/
//  {
//    Kokkos::deep_copy( aValues, mValues );
//  }

  /******************************************************************************/
  void get(Plato::ScalarVectorT<ScalarType> aValues, int aOffset=0, int aStride=1)
  /******************************************************************************/
  {
    TEUCHOS_TEST_FOR_EXCEPTION(
      mValues.extent(0)*aStride != aValues.extent(0),
      std::logic_error,
      "Size mismatch in field initialization:  Mod(view, stride) != 0");
    auto tFromValues = mValues;
    auto tToValues = aValues;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tFromValues.extent(0)), KOKKOS_LAMBDA(Plato::OrdinalType aPointOrdinal)
    {
        tToValues(aStride*aPointOrdinal+aOffset) = tFromValues(aPointOrdinal, 0);
    }, "copy");
  }

  /******************************************************************************/
  const decltype(mName)& name() { return mName; }
  /******************************************************************************/

}; // end class BodyLoad


/******************************************************************************/
/*!
  \brief Owner class that contains a vector of ComputeField instances
*/
template<int SpaceDim, typename ScalarType=Plato::Scalar>
class ComputedFields
/******************************************************************************/
{
  private:
    std::vector<std::shared_ptr<ComputedField<SpaceDim,ScalarType>>> CFs;

  public :

  /****************************************************************************/
  /*!
    \brief Constructor that parses and creates a vector of ComputedField instances
    based on the ParameterList.
  */
  ComputedFields(const Plato::Mesh aMesh, Teuchos::ParameterList &params) : CFs()
  /****************************************************************************/
  {
    for (Teuchos::ParameterList::ConstIterator i = params.begin(); i != params.end(); ++i)
    {
        const Teuchos::ParameterEntry &entry = params.entry(i);
        const std::string             &name  = params.name(i);

        TEUCHOS_TEST_FOR_EXCEPTION(!entry.isList(), std::logic_error, "Parameter in Computed Fields block not valid.  Expect lists only.");

        std::string function = params.sublist(name).get<std::string>("Function");
        auto newCF = std::make_shared<Plato::ComputedField<SpaceDim, ScalarType>>(aMesh, name, function);
        CFs.push_back(newCF);
    }
  }

  /****************************************************************************/
  /*!
    \brief Get the values for the specified field.
    \param aName Name of the requested Computed Field.
    \param aValues Computed Field values.
  */
  void get(const std::string& aName, Plato::ScalarVector& aValues)
  /****************************************************************************/
  {
      for (auto& cf : CFs) {
        if( cf->name() == aName ){
          cf->get(aValues);
          return;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Requested a Computed Field that doesn't exist.");
  }
  /****************************************************************************/
  /*!
    \brief Get the values for the specified field.
    Copies values from the computed field(iValue) into aValues(aStride*iValue+aOffset)
    \param aName Name of the requested Computed Field.
    \param aOffset Index of this degree of freedom
    \param aStride Number of degrees of freedom
    \param aValues Computed Field values.
  */
  void get(const std::string& aName, int aOffset, int aStride, Plato::ScalarVector& aValues)
  /****************************************************************************/
  {
      for (auto& cf : CFs) {
        if( cf->name() == aName ){
          cf->get(aValues, aOffset, aStride);
          return;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Requested a Computed Field that doesn't exist.");
  }
  /****************************************************************************/
  /*!
    \brief Find a Computed Field with the given name.
    \param aName Name of the requested Computed Field.
    This is a canary function.  If it doesn't find the requested Computed field a
     signal is thrown.
  */
  void find(const std::string& aName)
  /****************************************************************************/
  {
      for (auto& cf : CFs) {
        if( cf->name() == aName ){
          return;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Requested a Computed Field that doesn't exist.");
  }
};
} // end Plato namespace
#endif
