#ifndef PLATO_MECHANICS_HPP
#define PLATO_MECHANICS_HPP

#include <memory>

#include "parabolic/AbstractScalarFunction.hpp"

#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/ElastostaticResidual.hpp"
#include "elliptic/StressPNorm.hpp"
#include "elliptic/EffectiveEnergy.hpp"
#include "elliptic/Volume.hpp"
#include "elliptic/VolumeIntegralCriterion.hpp"
#include "elliptic/VolumeAverageCriterionDenominator.hpp"
#include "TensileEnergyDensityLocalMeasure.hpp"
#include "VonMisesLocalMeasure.hpp"
#include "elliptic/VolAvgStressPNormDenominator.hpp"
#include "Plato_AugLagStressCriterion.hpp"
#include "Plato_AugLagStressCriterionGeneral.hpp"
#include "Plato_AugLagStressCriterionQuadratic.hpp"
#include "AbstractLocalMeasure.hpp"
#include "IntermediateDensityPenalty.hpp"

#include "MakeFunctions.hpp"

#include "elliptic/InternalElasticEnergy.hpp"

#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace MechanicsFactory
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
      auto tLowerLocalMeasT = Plato::tolower(tLocalMeasure);
      if(tLowerLocalMeasT == "vonmises")
      {
          return std::make_shared<VonMisesLocalMeasure<EvaluationType>>
              (aSpatialDomain, aDataMap, aProblemParams, "VonMises");
      }
      else if(tLowerLocalMeasT == "tensileenergydensity")
      {
          return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType>>
              (aSpatialDomain, aDataMap, aProblemParams, "TensileEnergyDensity");
      }
      else
      {
          ANALYZE_THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
      }
  }


/******************************************************************************//**
 * \brief Create augmented Lagrangian stress constraint criterion tailored for linear problems
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_linear(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterion<EvaluationType> >
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    return (tOutput);
}

/******************************************************************************//**
 * \brief Create augmented Lagrangian stress constraint criterion tailored for general problems
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_general(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared <Plato::AugLagStressCriterionGeneral<EvaluationType> >
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    return (tOutput);
}


/******************************************************************************//**
 * \brief Create augmented Lagrangian local constraint criterion with quadratic constraint formulation
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_quadratic(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    auto EvalMeasure = Plato::MechanicsFactory::create_local_measure<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    using Residual = typename Plato::Elliptic::ResidualTypes<typename EvaluationType::ElementType>;
    auto PODMeasure = Plato::MechanicsFactory::create_local_measure<Residual>(aSpatialDomain, aDataMap, aProblemParams, aFuncName);

    std::shared_ptr<Plato::AugLagStressCriterionQuadratic<EvaluationType>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterionQuadratic<EvaluationType> >
        (aSpatialDomain, aDataMap, aProblemParams, aFuncName);

    tOutput->setLocalMeasure(EvalMeasure, PODMeasure);
    return (tOutput);
}

/******************************************************************************//**
 * \brief Create the numerator of the volume average criterion (i.e. a volume integral criterion)
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
volume_integral_criterion_for_volume_average(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    auto tLocalMeasure = Plato::MechanicsFactory::create_local_measure<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFuncName);

    std::shared_ptr<Plato::Elliptic::VolumeIntegralCriterion<EvaluationType>> tOutput;
    tOutput = std::make_shared<Plato::Elliptic::VolumeIntegralCriterion<EvaluationType>>
        (aSpatialDomain, aDataMap, aProblemParams, aFuncName);

    tOutput->setVolumeIntegratedQuantity(tLocalMeasure);
    return (tOutput);
}




/******************************************************************************//**
 * \brief Create volume average criterion denominator
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncName vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
vol_avg_criterion_denominator(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aFuncName
)
{
    return std::make_shared<Plato::Elliptic::VolumeAverageCriterionDenominator<EvaluationType>>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
}
// function vol_avg_criterion_denominator




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
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              std::string              aPDE)
    {
        auto tLowerPDE = Plato::tolower(aPDE);
        if(tLowerPDE == "elliptic")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::ElastostaticResidual>
                     (aSpatialDomain, aDataMap, aProblemParams, aPDE);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList")
        }
    }


    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
    createScalarFunctionParabolic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              strScalarFunctionType,
              std::string              aStrScalarFunctionName )
    /******************************************************************************/
    {
        ANALYZE_THROWERR("Not yet implemented")
    }

    /******************************************************************************//**
     * \brief Create a PLATO scalar function (i.e. optimization criterion)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Analyze physics-based database
     * \param [in] aProblemParams input parameters
     * \param [in] aFuncType scalar function type
     * \param [in] aFuncName scalar function name
    **********************************************************************************/
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
        if(tLowerFuncType == "internal elastic energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::InternalElasticEnergy>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress p-norm")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::StressPNorm>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "effective energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::EffectiveEnergy>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "volume")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::Volume>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if (tLowerFuncType == "volume average criterion numerator")
        {
            return Plato::MechanicsFactory::volume_integral_criterion_for_volume_average<EvaluationType>
                       (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if (tLowerFuncType == "volume average criterion denominator")
        {
            return Plato::MechanicsFactory::vol_avg_criterion_denominator<EvaluationType>
                       (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "vol avg stress p-norm denominator")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::VolAvgStressPNormDenominator>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress constraint")
        {
            return Plato::MechanicsFactory::stress_constraint_linear<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress constraint general")
        {
            return Plato::MechanicsFactory::stress_constraint_general<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress constraint quadratic")
        {
            return Plato::MechanicsFactory::stress_constraint_quadratic<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "density penalty")
        {
            return std::make_shared<Plato::IntermediateDensityPenalty<EvaluationType>>
                       (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
            const std::string tErrorString = std::string("Function '") + tLowerFuncType + "' not implemented yet in steady state mechanics.";
            ANALYZE_THROWERR(tErrorString)
        }
    }
};
// struct FunctionFactory

} // namespace MechanicsFactory

} // namespace Plato

#include "MechanicsElement.hpp"

namespace Plato
{
/******************************************************************************//**
 * \brief Concrete class for use as the Physics template argument in
 *        Plato::Elliptic::Problem
**********************************************************************************/
template<typename TopoElementType>
class Mechanics
{
public:
    typedef Plato::MechanicsFactory::FunctionFactory FunctionFactory;
    using ElementType = MechanicsElement<TopoElementType>;
};
} // namespace Plato

#endif
