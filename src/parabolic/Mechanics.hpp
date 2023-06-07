#ifndef PLATO_MECHANICS_HPP
#define PLATO_MECHANICS_HPP

#include <memory>

#include "MakeFunctions.hpp"
#include "AnalyzeMacros.hpp"

#include "parabolic/AbstractScalarFunction.hpp"
#include "parabolic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Parabolic
{

namespace LinearMechanics
{
  
/******************************************************************************//**
 * \brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
  /******************************************************************************//**
   * \brief Create a PLATO vector function (i.e. residual equation)
   * \param [in] aSpatialDomain Plato Analyze spatial domain
   * \param [in] aDataMap Plato Analyze physics-based database
   * \param [in] aProblemParams input parameters
   * \param [in] aPDE PDE type
  **********************************************************************************/
  template<typename EvaluationType>
  std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>>
  createVectorFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap, 
            Teuchos::ParameterList & aProblemParams,
            std::string              aPDE)
  {
    ANALYZE_THROWERR("Not yet implemented")
  }  

  template <typename EvaluationType>
  std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
  createScalarFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aParamList,
            std::string              strScalarFunctionType,
            std::string              aStrScalarFunctionName )
  /******************************************************************************/
  {
    ANALYZE_THROWERR("Not yet implemented")
  }
};
// struct FunctionFactory

} // namespace LinearMechanics

} // namespace Parabolic

} // namespace Plato

#include "MechanicsElement.hpp"

namespace Plato
{

namespace Parabolic
{

namespace Linear
{
  
/// @brief concrete class use to define parabolic mechanical physics
/// @tparam TopoElementType topological element typename
template<typename TopoElementType>
class Mechanics
{
public:
  /// @brief residual and criteria factory for parabolic mechanical physics
  typedef Plato::Parabolic::LinearMechanics::FunctionFactory FunctionFactory;
  /// @brief physics-based topological element typename
  using ElementType = MechanicsElement<TopoElementType>;
};

} // namespace Linear

} // namespace Parabolic

} // namespace Plato

#endif
