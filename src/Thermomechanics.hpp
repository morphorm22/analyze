#ifndef PLATO_THERMOMECHANICS_HPP
#define PLATO_THERMOMECHANICS_HPP

#include <memory>

#ifdef PLATO_PARABOLIC
#include "parabolic/TransientThermomechResidual.hpp"
#include "parabolic/AbstractScalarFunction.hpp"
#include "parabolic/InternalThermoelasticEnergy.hpp"
#endif

#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/ThermoelastostaticResidual.hpp"
#include "elliptic/InternalThermoelasticEnergy.hpp"
#include "elliptic/TMStressPNorm.hpp"
#include "elliptic/Volume.hpp"


#include "AbstractLocalMeasure.hpp"

#include "AnalyzeMacros.hpp"

#include "Plato_AugLagStressCriterionQuadratic.hpp"
#include "ThermalVonMisesLocalMeasure.hpp"

namespace Plato
{

namespace ThermomechanicsFactory
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
            Teuchos::ParameterList & aProblemParams,
      const std::string            & aFuncName
    )
    {
        auto tFunctionSpecs = aProblemParams.sublist("Criteria").sublist(aFuncName);
        auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

        if(tLocalMeasure == "VonMises")
        {
          return std::make_shared<ThermalVonMisesLocalMeasure<EvaluationType>>
              (aSpatialDomain, aProblemParams, "VonMises");
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
    inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    stress_constraint_quadratic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName)
    {
        auto EvalMeasure = Plato::ThermomechanicsFactory::create_local_measure<EvaluationType>(aSpatialDomain, aInputParams, aFuncName);
        using Residual = typename Plato::Elliptic::ResidualTypes<typename EvaluationType::ElementType>;
        auto PODMeasure = Plato::ThermomechanicsFactory::create_local_measure<Residual>(aSpatialDomain, aInputParams, aFuncName);

        std::shared_ptr<Plato::AugLagStressCriterionQuadratic<EvaluationType>> tOutput;
        tOutput = std::make_shared< Plato::AugLagStressCriterionQuadratic<EvaluationType> >
                    (aSpatialDomain, aDataMap, aInputParams, aFuncName);

        tOutput->setLocalMeasure(EvalMeasure, PODMeasure);
        return (tOutput);
    }

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
            return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::ThermoelastostaticResidual>
                     (aSpatialDomain, aDataMap, aParamList, aFuncType);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

#ifdef PLATO_PARABOLIC
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>>
    createVectorFunctionParabolic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              strVectorFunctionType
    )
    /******************************************************************************/
    {
        if( strVectorFunctionType == "Parabolic" )
        {
            auto tPenaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type");
            if( tPenaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Parabolic::TransientThermomechResidual<EvaluationType, Plato::MSIMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            } else
            if( tPenaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Parabolic::TransientThermomechResidual<EvaluationType, Plato::RAMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            } else
            if( tPenaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Parabolic::TransientThermomechResidual<EvaluationType, Plato::Heaviside>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            } else {
                ANALYZE_THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
#endif

    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              std::string              aFuncType,
              std::string              aFuncName
    )
    /******************************************************************************/
    {

        auto tLowerFuncType = Plato::tolower(aFuncType);
        if(tLowerFuncType == "internal thermoelastic energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::InternalThermoelasticEnergy>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else 
        if(tLowerFuncType == "stress p-norm")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::TMStressPNorm>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        if(tLowerFuncType == "stress constraint quadratic")
        {
            return (Plato::ThermomechanicsFactory::stress_constraint_quadratic<EvaluationType>
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

#ifdef PLATO_PARABOLIC
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
    createScalarFunctionParabolic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              aStrScalarFunctionType,
              std::string              aStrScalarFunctionName
    )
    /******************************************************************************/
    {
        auto tPenaltyParams = aParamList.sublist("Criteria").sublist(aStrScalarFunctionName).sublist("Penalty Function");
        std::string tPenaltyType = tPenaltyParams.get<std::string>("Type");
        if( aStrScalarFunctionType == "Internal Thermoelastic Energy" )
        {
            if( tPenaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Parabolic::InternalThermoelasticEnergy<EvaluationType, Plato::MSIMP>>
                     (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if( tPenaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Parabolic::InternalThermoelasticEnergy<EvaluationType, Plato::RAMP>>
                     (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if( tPenaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Parabolic::InternalThermoelasticEnergy<EvaluationType, Plato::Heaviside>>
                     (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                ANALYZE_THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
#endif

}; // struct FunctionFactory

} // namespace ThermomechanicsFactory

} // namespace Plato

#include "ThermomechanicsElement.hpp"

namespace Plato
{
/****************************************************************************//**
 * \brief Concrete class for use as the SimplexPhysics template argument in
 *        Plato::Elliptic::Problem and Plato::Parabolic::Problem
 *******************************************************************************/
template<typename TopoElementType>
class Thermomechanics
{
public:
    typedef Plato::ThermomechanicsFactory::FunctionFactory FunctionFactory;
    using ElementType = ThermomechanicsElement<TopoElementType>;
};
} // namespace Plato

#endif
