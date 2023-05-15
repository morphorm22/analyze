#ifndef LINEARTHERMALMATERIAL_HPP
#define LINEARTHERMALMATERIAL_HPP

#include <Teuchos_ParameterList.hpp>
#include "MaterialModel.hpp"

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Base class for Linear Thermal material models
 */
template<typename EvaluationType>
class ThermalConductionModel : public MaterialModel<EvaluationType>
/******************************************************************************/
{
  public:
    ThermalConductionModel(const Teuchos::ParameterList& paramList);
};

/******************************************************************************/
template<typename EvaluationType>
ThermalConductionModel<EvaluationType>::
ThermalConductionModel(const Teuchos::ParameterList& paramList) : MaterialModel<EvaluationType>(paramList)
/******************************************************************************/
{
    this->parseTensor("Thermal Conductivity", paramList);
}

/******************************************************************************/
/*!
 \brief Factory for creating material models
 */
template<typename EvaluationType>
class ThermalConductionModelFactory
/******************************************************************************/
{
public:
    ThermalConductionModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList)
    {
    }
    Teuchos::RCP<MaterialModel<EvaluationType>> create(std::string aModelName);
private:
    const Teuchos::ParameterList& mParamList;
};
/******************************************************************************/
template<typename EvaluationType>
Teuchos::RCP<MaterialModel<EvaluationType>>
ThermalConductionModelFactory<EvaluationType>::create(std::string aModelName)
/******************************************************************************/
{
    if (!mParamList.isSublist("Material Models"))
    {
        REPORT("'Material Models' list not found! Returning 'nullptr'");
        return Teuchos::RCP<Plato::MaterialModel<EvaluationType>>(nullptr);
    }
    else
    {
        auto tModelsParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");

        if (!tModelsParamList.isSublist(aModelName))
        {
            std::stringstream ss;
            ss << "Requested a material model ('" << aModelName << "') that isn't defined";
            ANALYZE_THROWERR(ss.str());
        }

        auto tModelParamList = tModelsParamList.sublist(aModelName);
        if(tModelParamList.isSublist("Thermal Conduction"))
        {
            return Teuchos::rcp(new ThermalConductionModel<EvaluationType>(
                tModelParamList.sublist("Thermal Conduction")));
        }
        else
          ANALYZE_THROWERR("Expected 'Thermal Conduction' ParameterList");
    }
}

}

#endif
