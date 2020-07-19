#ifndef PLATO_HYPERBOLIC_MECHANICS_HPP
#define PLATO_HYPERBOLIC_MECHANICS_HPP

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "SimplexMechanics.hpp"
#include "hyperbolic/HyperbolicAbstractScalarFunction.hpp"
#include "hyperbolic/ElastomechanicsResidual.hpp"
#include "hyperbolic/HyperbolicInternalElasticEnergy.hpp"
#include "hyperbolic/HyperbolicStressPNorm.hpp"

namespace Plato
{
  namespace Hyperbolic
  {
    struct FunctionFactory
    {
      /******************************************************************************/
      template <typename EvaluationType>
      std::shared_ptr<::Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>>
      createVectorFunctionHyperbolic(
          Omega_h::Mesh          & aMesh,
          Omega_h::MeshSets      & aMeshSets,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string              strVectorFunctionType
      )
      {
        auto tFunctionParams = aProblemParams.sublist(strVectorFunctionType);
        if( strVectorFunctionType == "Hyperbolic" )
        {
            std::string tPenaltyType = tFunctionParams.sublist("Penalty Function").get<std::string>("Type");
            if( tPenaltyType == "SIMP" )
            {
                return std::make_shared<TransientMechanicsResidual<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tFunctionParams);
            } else
            if( tPenaltyType == "RAMP" )
            {
                return std::make_shared<TransientMechanicsResidual<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tFunctionParams);
            } else
            if( tPenaltyType == "Heaviside" )
            {
                return std::make_shared<TransientMechanicsResidual<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tFunctionParams);
            } else {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
      }
      /******************************************************************************/
      template <typename EvaluationType>
      std::shared_ptr<::Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>>
      createScalarFunction(
        Omega_h::Mesh&          aMesh,
        Omega_h::MeshSets&      aMeshSets,
        Plato::DataMap&         aDataMap,
        Teuchos::ParameterList& aProblemParams,
        std::string             strScalarFunctionType,
        std::string             strScalarFunctionName
      )
      /******************************************************************************/
      {
        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(strScalarFunctionName);
        std::string tPenaltyType = tFunctionParams.sublist("Penalty Function").get<std::string>("Type");

        if( strScalarFunctionType == "Internal Elastic Energy" )
        {
          if( tPenaltyType == "SIMP" ){
            return std::make_shared<Plato::Hyperbolic::InternalElasticEnergy<EvaluationType, Plato::MSIMP>>
                     (aMesh, aMeshSets, aDataMap, aProblemParams, tFunctionParams, strScalarFunctionName);
          } else
          if( tPenaltyType == "RAMP" ){
            return std::make_shared<Plato::Hyperbolic::InternalElasticEnergy<EvaluationType, Plato::RAMP>>
                     (aMesh, aMeshSets, aDataMap, aProblemParams, tFunctionParams, strScalarFunctionName);
          } else
          if( tPenaltyType == "Heaviside" ){
            return std::make_shared<Plato::Hyperbolic::InternalElasticEnergy<EvaluationType, Plato::Heaviside>>
                     (aMesh, aMeshSets, aDataMap, aProblemParams, tFunctionParams, strScalarFunctionName);
          } else {
            throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
          }
        }
        else
        if( strScalarFunctionType == "Stress P-Norm" )
        {
          if( tPenaltyType == "SIMP" ){
            return std::make_shared<Plato::Hyperbolic::StressPNorm<EvaluationType, Plato::MSIMP>>
                     (aMesh, aMeshSets, aDataMap, aProblemParams, tFunctionParams, strScalarFunctionName);
          } else
          if( tPenaltyType == "RAMP" ){
            return std::make_shared<Plato::Hyperbolic::StressPNorm<EvaluationType, Plato::RAMP>>
                     (aMesh,aMeshSets,aDataMap, aProblemParams, tFunctionParams, strScalarFunctionName);
          } else
          if( tPenaltyType == "Heaviside" ){
            return std::make_shared<Plato::Hyperbolic::StressPNorm<EvaluationType, Plato::Heaviside>>
                     (aMesh, aMeshSets, aDataMap, aProblemParams, tFunctionParams, strScalarFunctionName);
          } else {
            throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
          }
        }
        else
        {
          throw std::runtime_error("Unknown 'Criterion' specified in 'Plato Problem' ParameterList");
        }
      }
    };

    /******************************************************************************//**
     * @brief Concrete class for use as the SimplexPhysics template argument in
     *        Plato::Hyperbolic::Problem
    **********************************************************************************/
    template<Plato::OrdinalType SpaceDimParam>
    class Mechanics: public Plato::SimplexMechanics<SpaceDimParam>
    {
    public:
        typedef Plato::Hyperbolic::FunctionFactory FunctionFactory;
        using SimplexT = SimplexMechanics<SpaceDimParam>;
        static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
    };
  } // namespace Hyperbolic

} // namespace Plato

#endif
