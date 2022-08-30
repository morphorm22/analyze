#ifndef PLATO_THERMOPLASTICITY_HPP
#define PLATO_THERMOPLASTICITY_HPP

#include <memory>

#include "AnalyzeMacros.hpp"
#include "SimplexThermoPlasticity.hpp"
#include "J2PlasticityLocalResidual.hpp"

#ifdef PLATO_CUSTOM_MATERIALS
  #include "J2PlasticityLocalResidualExpFAD.hpp"
#else
  #include "J2PlasticityLocalResidual.hpp"
#endif

namespace Plato
{

namespace ThermoPlasticityFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO local vector function  inc (i.e. local residual equations)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze physics-based database
     * \param [in] aInputParams input parameters
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<EvaluationType>>
    createLocalVectorFunctionInc(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams
    )
    {
        if(aInputParams.isSublist("Material Models") == false)
        {
            ANALYZE_THROWERR("'Material Models' Sublist is not defined.")
        }
        Teuchos::ParameterList tMaterialModelsList = aInputParams.sublist("Material Models");
        Teuchos::ParameterList tMaterialModelList  = tMaterialModelsList.sublist(aSpatialDomain.getMaterialName());

        if(tMaterialModelList.isSublist("Plasticity Model") == false)
        {
            ANALYZE_THROWERR("Plasticity Model Sublist is not defined.")
        }

        auto tPlasticityParamList = tMaterialModelList.get<Teuchos::ParameterList>("Plasticity Model");

        if(tPlasticityParamList.isSublist("J2 Plasticity"))
        {
          constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
          return std::make_shared
            <J2PlasticityLocalResidual<EvaluationType, Plato::SimplexThermoPlasticity<tSpaceDim>>>
            (aSpatialDomain, aDataMap, aInputParams);
        }
        else
        {
          const std::string tError = std::string("Unknown Plasticity Model.  Options are: J2 Plasticity.")
              + "User is advised to select one of the available options.";
          ANALYZE_THROWERR(tError)
        }
    }
}; // struct FunctionFactory

} // namespace ThermoPlasticityFactory


/****************************************************************************//**
 * \brief Concrete class for use as the PhysicsT template argument in VectorFunctionVMS
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class ThermoPlasticity: public Plato::SimplexThermoPlasticity<SpaceDimParam>
{
public:
    typedef Plato::ThermoPlasticityFactory::FunctionFactory FunctionFactory;
    using SimplexT        = Plato::SimplexThermoPlasticity<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

} // namespace Plato

#endif
