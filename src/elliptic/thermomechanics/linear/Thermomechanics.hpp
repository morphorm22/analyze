#pragma once

#include <memory>

#include "base/ResidualBase.hpp"
#include "base/CriterionBase.hpp"

#include "elliptic/Volume.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/thermomechanics/linear/ResidualThermoelastostatic.hpp"
#include "elliptic/thermomechanics/linear/CriterionInternalThermoelasticEnergy.hpp"
#include "elliptic/thermomechanics/linear/CriterionThermoMechStressPNorm.hpp"
#include "elliptic/thermomechanics/linear/LocalMeasureThermalVonMises.hpp"

#include "elliptic/AbstractLocalMeasure.hpp"
#include "elliptic/mechanical/linear/Plato_AugLagStressCriterionQuadratic.hpp"

#include "MakeFunctions.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

namespace LinearThermoMechanics
{
  
/******************************************************************************//**
* \brief Create a local measure for use in augmented lagrangian quadratic
* \param [in] aProblemParams input parameters
* \param [in] aFuncName scalar function name
**********************************************************************************/
template <typename EvaluationType>
inline std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>>
create_local_measure(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
  const std::string            & aFuncName
)
{
  auto tFunctionSpecs = aProblemParams.sublist("Criteria").sublist(aFuncName);
  auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");  
  if(tLocalMeasure == "VonMises")
  {
    return std::make_shared<LocalMeasureThermalVonMises<EvaluationType>>
        (aSpatialDomain, aDataMap, aProblemParams, "VonMises");
  }
  else
  {
    ANALYZE_THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
  }
}

/******************************************************************************//**
 * \brief Create augmented Lagrangian local constraint criterion with quadratic constraint formulation
 * \param [in] aMesh mesh database
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::CriterionBase>
stress_constraint_quadratic(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
  const std::string            & aFuncName)
{
  auto EvalMeasure = Plato::Elliptic::LinearThermoMechanics::create_local_measure<EvaluationType>(
    aSpatialDomain, aDataMap, aInputParams, aFuncName);
  using Residual = typename Plato::Elliptic::ResidualTypes<typename EvaluationType::ElementType>;
  auto PODMeasure = Plato::Elliptic::LinearThermoMechanics::create_local_measure<Residual>(
    aSpatialDomain, aDataMap, aInputParams, aFuncName);  
  auto tOutput = std::make_shared< Plato::AugLagStressCriterionQuadratic<EvaluationType> >
              (aSpatialDomain, aDataMap, aInputParams, aFuncName);  
  tOutput->setLocalMeasure(EvalMeasure, PODMeasure);
  return (tOutput);
}

struct FunctionFactory
{
  /******************************************************************************/
  template<typename EvaluationType>
  std::shared_ptr<Plato::ResidualBase>
  createVectorFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList,
          std::string              aFuncType
  )
  {  
    auto tLowerFuncType = Plato::tolower(aFuncType);
    if(tLowerFuncType == "elliptic")
    {
      return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::ResidualThermoelastostatic>
               (aSpatialDomain, aDataMap, aParamList, aFuncType);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }

  template<typename EvaluationType>
  std::shared_ptr<Plato::CriterionBase>
  createScalarFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap, 
          Teuchos::ParameterList & aProblemParams, 
          std::string              aFuncType,
          std::string              aFuncName
  )
  {  
    auto tLowerFuncType = Plato::tolower(aFuncType);
    if(tLowerFuncType == "internal thermoelastic energy")
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::CriterionInternalThermoelasticEnergy>
          (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else 
    if(tLowerFuncType == "stress p-norm")
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::CriterionThermoMechStressPNorm>
          (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else
    if(tLowerFuncType == "stress constraint quadratic")
    {
      return (Plato::Elliptic::LinearThermoMechanics::stress_constraint_quadratic<EvaluationType>
             (aSpatialDomain, aDataMap, aProblemParams, aFuncName));
    }
    else
    if(tLowerFuncType == "volume" )
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::Volume>
          (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
    }
  }

}; // struct FunctionFactory

} // namespace LinearThermoMechanics

} // namespace Elliptic

} // namespace Plato

#include "ThermomechanicsElement.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Linear
{
  
/// @brief concrete class use to define elliptic thermomechanics physics
/// @tparam TopoElementType topological element typename 
template<typename TopoElementType>
class Thermomechanics : public Plato::ThermomechanicsElement<TopoElementType>
{
public:
  /// @brief residual and criteria factory for elliptic thermomechanics physics
  typedef Plato::Elliptic::LinearThermoMechanics::FunctionFactory FunctionFactory;
  /// @brief topological element type with additional physics related information 
  using ElementType = ThermomechanicsElement<TopoElementType>;
};

} // namespace Linear

} // namespace Elliptic

} // namespace Plato
