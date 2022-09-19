#ifndef PLATO_PARSE_TOOLS
#define PLATO_PARSE_TOOLS

#include "PlatoTypes.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoMathTypes.hpp"
#include "Teuchos_ParameterList.hpp"

#include <sstream>
#include <string>

namespace Plato {

namespace ParseTools {

/**************************************************************************//**
 * \brief Get a parameter from a sublist if it exists, otherwise return the default.
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aSubListName The name of the sublist within aInputParams
 * \param [in] aParamName The name of the desired parameter
 * \return The requested parameter value if it exists, otherwise the default
 *****************************************************************************/

template < typename T >
T getSubParam(
    Teuchos::ParameterList& aInputParams,
    const std::string aSubListName,
    const std::string aParamName,
    T aDefaultValue )
{
    if( aInputParams.isSublist(aSubListName) == true )
    {
        return aInputParams.sublist(aSubListName).get<T>(aParamName, aDefaultValue);
    }
    else
    {
        return aDefaultValue;
    }
}

/**************************************************************************//**
 * \brief Get a parameter if it exists, otherwise return the default.
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aParamName The name of the desired parameter
 * \param [in] aDefaultValue The default value
 * \return The requested parameter value if it exists, otherwise the default
 *****************************************************************************/

template < typename T >
T getParam(
    Teuchos::ParameterList& aInputParams,
    const std::string aParamName,
    T aDefaultValue )
{
    if (aInputParams.isType<T>(aParamName))
    {
        return aInputParams.get<T>(aParamName);
    }
    else
    {
        return aDefaultValue;
    }
}

/**************************************************************************//**
 * \brief Get a parameter if it exists, otherwise throw an exception
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aParamName The name of the desired parameter
 * \return The requested parameter value if it exists, otherwise throw
 *****************************************************************************/

template < typename T >
T getParam(
    const Teuchos::ParameterList& aInputParams,
    const std::string aParamName )
{
    if (aInputParams.isType<T>(aParamName))
    {
        return aInputParams.get<T>(aParamName);
    }
    else
    {
        std::stringstream sstream;
        sstream << "Missing required parameter " << aParamName << std::endl;
        ANALYZE_THROWERR(sstream.str());
    }
}

/**************************************************************************//**
 * \brief Verify that the input vector is the specified length. Throw an
 *        exception if not.
 * \tparam T Type that the array contains
 * \param [in] aVector Vector to check
 * \param [in] aLength Expected length of the input vector
 *****************************************************************************/
template < typename T >
void
verifyVectorLength(Teuchos::Array<T> aVector, int aLength, std::string aContext="")
{
  if( aVector.size() != aLength )
  {
      std::stringstream ss;
      ss << aContext << std::endl << "Incorrect vector length:" << aVector << std::endl;
      ss << "Expected length: " << aLength << ", but found length: " << aVector.size() << std::endl;
      ANALYZE_THROWERR(ss.str());
  }
}

/**************************************************************************//**
 * \brief Normalize the input vector
 * \tparam T Type that the array contains
 * \param [in/out] aVector
 *****************************************************************************/
template < typename T >
void
normalizeVector(Teuchos::Array<T> & aVector, bool aQuiet=false, std::string aContext="")
{
  T tMag(0.0);
  for(auto tVal : aVector)
  {
    tMag += tVal*tVal;
  }

  if( tMag == 0 )
  {
    ANALYZE_THROWERR("Attempted to normalize a vector of length zero.");
  }

  tMag = sqrt(tMag);

  T tOne(1.0);
  if(fabs(tOne - tMag) > DBL_EPSILON)
  {
    if(!aQuiet)
    {
      std::stringstream ss;
      ss << aContext << std::endl << "Normalizing vector" << aVector << std::endl;
      REPORT(ss.str());
    }
    for(auto & tVal : aVector)
    {
      tVal /= tMag;
    }
  }
}


/**************************************************************************//**
 * \brief Parse a 3D cartesian basis if it exists, otherwise throw an exception
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aParamName The name of the desired parameter
 * \return The requested parameter value if it exists, otherwise throw
 *****************************************************************************/
template < typename T >
void
getBasis(
    Teuchos::ParameterList const & aParams,
    Plato::Matrix<3, 3, T>       & aBasis
)
{
  if (aParams.isSublist("Basis"))
  {
    auto tBasisList = aParams.sublist("Basis");

    auto tXbasis = Plato::ParseTools::getParam<Teuchos::Array<T>>(tBasisList, "X");
    verifyVectorLength(tXbasis, 3);
    normalizeVector(tXbasis, /*quiet=*/false, "Reading basis");
 
    auto tYbasis = Plato::ParseTools::getParam<Teuchos::Array<T>>(tBasisList, "Y");
    verifyVectorLength(tYbasis, 3);
    normalizeVector(tYbasis, /*quiet=*/false, "Reading basis");

    // Z basis vector is optional.  If not provided, compute as cross product of X and Y.
    Teuchos::Array<T> tZbasis(3);
    if(tBasisList.isSublist("Z"))
    {
      tZbasis = Plato::ParseTools::getParam<Teuchos::Array<T>>(tBasisList, "Z");
      verifyVectorLength(tZbasis, 3);
      normalizeVector(tXbasis, /*quiet=*/false, "Reading basis");
    }
    else
    {
      tZbasis[0] = tXbasis[1]*tYbasis[2] - tXbasis[2]*tYbasis[1];
      tZbasis[1] = tXbasis[2]*tYbasis[0] - tXbasis[0]*tYbasis[2];
      tZbasis[2] = tXbasis[0]*tYbasis[1] - tXbasis[1]*tYbasis[0];
    }

    for(int i=0; i<3; i++)
    {
      aBasis(i,0) = tXbasis[i];
      aBasis(i,1) = tYbasis[i];
      aBasis(i,2) = tZbasis[i];
    }
  }
  else
  {
    ANALYZE_THROWERR("Attempted to parse a 'Basis' ParameterList that doesn't exist");
  }
}

/**************************************************************************//**
 * \brief Parse a 2D cartesian basis if it exists, otherwise throw an exception
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aParamName The name of the desired parameter
 * \return The requested parameter value if it exists, otherwise throw
 *****************************************************************************/
template < typename T >
void
getBasis(
    Teuchos::ParameterList const & aParams,
    Plato::Matrix<2, 2, T>       & aBasis
)
{
  if (aParams.isSublist("Basis"))
  {
    auto tBasisList = aParams.sublist("Basis");

    auto tXbasis = Plato::ParseTools::getParam<Teuchos::Array<T>>(tBasisList, "X");
    verifyVectorLength(tXbasis, 2);
    normalizeVector(tXbasis, /*quiet=*/false, "Reading basis");
 
    // Y basis vector is optional.  If not provided, compute from X.
    Teuchos::Array<T> tYbasis(2);
    if(tBasisList.isSublist("Y"))
    {
      tYbasis = Plato::ParseTools::getParam<Teuchos::Array<T>>(tBasisList, "Y");
      verifyVectorLength(tYbasis, 2);
      normalizeVector(tXbasis, /*quiet=*/false, "Reading basis");
    }
    else
    {
      tYbasis[0] = -tXbasis[1];
      tYbasis[1] =  tXbasis[0];
    }

    for(int i=0; i<2; i++)
    {
      aBasis(i,0) = tXbasis[i];
      aBasis(i,1) = tYbasis[i];
    }
  }
  else
  {
    ANALYZE_THROWERR("Attempted to parse a 'Basis' ParameterList that doesn't exist");
  }
}

/**************************************************************************//**
 * \brief Parse a 1D cartesian basis if it exists, otherwise throw an exception
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aParamName The name of the desired parameter
 * \return The requested parameter value if it exists, otherwise throw
 *****************************************************************************/
template < typename T >
void
getBasis(
    Teuchos::ParameterList const & aParams,
    Plato::Matrix<1, 1, T>       & aBasis
)
{
  if (aParams.isSublist("Basis"))
  {
    auto tBasisList = aParams.sublist("Basis");

    auto tXbasis = Plato::ParseTools::getParam<Teuchos::Array<T>>(tBasisList, "X");
    verifyVectorLength(tXbasis, 1);
    normalizeVector(tXbasis, /*quiet=*/false, "Reading basis");
  }
  else
  {
    ANALYZE_THROWERR("Attempted to parse a 'Basis' ParameterList that doesn't exist");
  }
}

/**************************************************************************//**
 * \brief Get an equation if it exists, otherwise throw an exception
 * \param [in] aInputParams The containing ParameterList
 * \param [in] equationName The name of the desired equation
 * \return The requested equation, otherwise throw
 *****************************************************************************/

std::string getEquationParam(const Teuchos::ParameterList& aInputParams,
                             const std::string equationName );

/**************************************************************************//**
 * \brief Get an equation if it exists, otherwise throw an exception
 * \param [in] aInputParams The containing ParameterList
 * \param [in] equationIndex The index of the desired equation in a Bingo File
 * \param [in] equationName The name of the desired equation
 * \return The requested equation, otherwise throw
 *****************************************************************************/

std::string getEquationParam(const Teuchos::ParameterList& aInputParams,
                             const Plato::OrdinalType equationIndex = -1,
                             const std::string equationName = std::string("Equation") );

} // namespace ParseTools

} // namespace Plato

#endif
