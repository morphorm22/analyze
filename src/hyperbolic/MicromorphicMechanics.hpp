#pragma once

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "SpatialModel.hpp"

#include "hyperbolic/SimplexMicromorphicMechanics.hpp"
#include "hyperbolic/RelaxedMicromorphicResidual.hpp"

namespace Plato
{
  namespace Hyperbolic
  {
    struct MicromorphicFunctionFactory
    {
      /******************************************************************************/
      template <typename EvaluationType>
      std::shared_ptr<::Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>>
      createVectorFunctionHyperbolic(
          const Plato::SpatialDomain   & aSpatialDomain,
                Plato::DataMap         & aDataMap,
                Teuchos::ParameterList & aProblemParams,
                std::string              strVectorFunctionType
      )
      {
        if( !aProblemParams.isSublist(strVectorFunctionType) )
        {
            std::cout << " Warning: '" << strVectorFunctionType << "' ParameterList not found" << std::endl;
            std::cout << " Warning: Using defaults. " << std::endl;
        }
        auto tFunctionParams = aProblemParams.sublist(strVectorFunctionType);
        if( strVectorFunctionType == "Hyperbolic" )
        {
            if( !tFunctionParams.isSublist("Penalty Function") )
            {
                std::cout << " Warning: 'Penalty Function' ParameterList not found" << std::endl;
                std::cout << " Warning: Using defaults. " << std::endl;
            }
            auto tPenaltyParams = tFunctionParams.sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type");
            if( tPenaltyType == "SIMP" )
            {
                std::cout << tFunctionParams << std::endl;
                return std::make_shared<RelaxedMicromorphicResidual<EvaluationType, Plato::MSIMP>>
                         (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
            } else
            if( tPenaltyType == "RAMP" )
            {
                return std::make_shared<RelaxedMicromorphicResidual<EvaluationType, Plato::RAMP>>
                         (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
            } else
            if( tPenaltyType == "Heaviside" )
            {
                return std::make_shared<RelaxedMicromorphicResidual<EvaluationType, Plato::Heaviside>>
                         (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
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
          const Plato::SpatialDomain   & aSpatialDomain,
                Plato::DataMap&          aDataMap,
                Teuchos::ParameterList & aProblemParams,
                std::string              strScalarFunctionType,
                std::string              strScalarFunctionName
      )
      /******************************************************************************/
      {
          THROWERR("No criteria supported for micromorphic mechanics currently.")
      }
    };

    /******************************************************************************//**
     * \brief Concrete class for use as the SimplexPhysics template argument in
     *        Plato::Hyperbolic::Problem
    **********************************************************************************/
    template<Plato::OrdinalType SpaceDimParam>
    class MicromorphicMechanics: public Plato::SimplexMicromorphicMechanics<SpaceDimParam>
    {
    public:
        typedef Plato::Hyperbolic::MicromorphicFunctionFactory FunctionFactory;
        using SimplexT = SimplexMicromorphicMechanics<SpaceDimParam>;
        static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
    };
  } // namespace Hyperbolic

} // namespace Plato

