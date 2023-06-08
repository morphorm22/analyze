#ifndef PLATO_ELECTROMECHANICS_HPP
#define PLATO_ELECTROMECHANICS_HPP

#include <memory>

#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/electromechanics/ElectroelastostaticResidual.hpp"
#include "elliptic/electromechanics/InternalElectroelasticEnergy.hpp"
#include "elliptic/electromechanics/EMStressPNorm.hpp"

#include "MakeFunctions.hpp"

namespace Plato
{

namespace Elliptic
{

namespace LinearElectroMechanics
{
  
/******************************************************************************/
struct FunctionFactory
{
  /******************************************************************************/
  template<typename EvaluationType>
  std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
  createVectorFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap, 
            Teuchos::ParameterList & aParamList,
            std::string              aFuncType
  )
  /******************************************************************************/
  {  
    auto tLowerFuncType = Plato::tolower(aFuncType);
    if(tLowerFuncType == "elliptic")
    {
      return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::ElectroelastostaticResidual>
               (aSpatialDomain, aDataMap, aParamList, aFuncType);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }

  template<typename EvaluationType>
  std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
  createScalarFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap, 
            Teuchos::ParameterList & aProblemParams, 
            std::string              aFuncType,
            std::string              aFuncName
  )
  {
    auto tLowerFuncType = Plato::tolower(aFuncType);
    if(tLowerFuncType == "internal electroelastic energy")
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::InternalElectroelasticEnergy>(
        aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else
    if(tLowerFuncType == "stress p-norm")
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::EMStressPNorm>(
        aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else
    {
      throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
    }
  }
}; // struct FunctionFactory

} // namespace LinearElectroMechanics

} // namespace Elliptic

} // namespace Plato

#include "ElectromechanicsElement.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Linear
{
  
/// @brief concrete class use to define elliptic electromechanics physics
/// @tparam TopoElementType topological element typename 
template<typename TopoElementType>
class Electromechanics : public Plato::ElectromechanicsElement<TopoElementType>
{
public:
  /// @brief residual and criteria factory for elliptic electromechanics physics 
  typedef Plato::Elliptic::LinearElectroMechanics::FunctionFactory FunctionFactory;
  /// @brief topological element type with additional physics related information 
  using ElementType = ElectromechanicsElement<TopoElementType>;
};

} // namespace Linear

} // namespace Elliptic

} // namespace Plato

#endif
