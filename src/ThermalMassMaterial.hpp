#pragma once

#include <Teuchos_ParameterList.hpp>
#include "PlatoStaticsTypes.hpp"
#include "materials/MaterialModel.hpp"

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Base class for ThermalMass material models
  */
    template<typename EvaluationType>
    class ThermalMassMaterial : public MaterialModel<EvaluationType>
  /******************************************************************************/
  {
  
    public:
      ThermalMassMaterial(const Teuchos::ParameterList& paramList);
  };

  /******************************************************************************/
  template<typename EvaluationType>
  ThermalMassMaterial<EvaluationType>::
  ThermalMassMaterial(const Teuchos::ParameterList& paramList) : 
    MaterialModel<EvaluationType>(paramList)
  /******************************************************************************/
  {
      this->parseScalar("Mass Density", paramList);
      this->parseScalar("Specific Heat", paramList);
  }

  /******************************************************************************/
  /*!
    \brief Factory for creating material models
  */
    template<typename EvaluationType>
    class ThermalMassModelFactory
  /******************************************************************************/
  {
    public:
      ThermalMassModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
      Teuchos::RCP<Plato::MaterialModel<EvaluationType>> create(std::string aModelName);
    private:
      const Teuchos::ParameterList& mParamList;
  };

  /******************************************************************************/
  template<typename EvaluationType>
  Teuchos::RCP<MaterialModel<EvaluationType>>
  ThermalMassModelFactory<EvaluationType>::create(std::string aModelName)
  /******************************************************************************/
  {
      if (!mParamList.isSublist("Material Models"))
      {
          REPORT("'Material Models' list not found! Returning 'nullptr'");
          return Teuchos::RCP<Plato::MaterialModel<EvaluationType>>(nullptr);
      }
      else
      {
          auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");

          if (!tModelsParamList.isSublist(aModelName))
          {
              std::stringstream ss;
              ss << "Requested a material model ('" << aModelName << "') that isn't defined";
              ANALYZE_THROWERR(ss.str());
          }

          auto tModelParamList = tModelsParamList.sublist(aModelName);

          if( tModelParamList.isSublist("Thermal Mass") )
          {
              return Teuchos::rcp(new Plato::ThermalMassMaterial<EvaluationType>
                     (tModelParamList.sublist("Thermal Mass")));
          }
          else
          {
              ANALYZE_THROWERR("Expected 'Thermal Mass' ParameterList");
          }
      }
  }
} // namespace Plato
