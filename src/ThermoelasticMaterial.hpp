#pragma once

#include <Teuchos_ParameterList.hpp>
#include "PlatoStaticsTypes.hpp"
#include "MaterialModel.hpp"

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Base class for Thermoelastic material models
  */
    template<int SpatialDim>
    class ThermoelasticMaterial : public MaterialModel<SpatialDim>
  /******************************************************************************/
  {
  
    public:
      ThermoelasticMaterial(const Teuchos::ParameterList& paramList);

    private:
      void parseElasticStiffness(const Teuchos::ParameterList& paramList);
  };

  /******************************************************************************/
  template<int SpatialDim>
  ThermoelasticMaterial<SpatialDim>::
  ThermoelasticMaterial(const Teuchos::ParameterList& paramList) : MaterialModel<SpatialDim>(paramList)
  /******************************************************************************/
  {
      this->parseElasticStiffness(paramList);
      this->parseTensor("Thermal Expansivity", paramList);
      this->parseTensor("Thermal Conductivity", paramList);

      this->parseScalarConstant("Reference Temperature", paramList, 23.0);
      this->parseScalarConstant("Temperature Scaling", paramList, 1.0);
  }

  /******************************************************************************/
  template<int SpatialDim>
  void ThermoelasticMaterial<SpatialDim>::
  parseElasticStiffness(const Teuchos::ParameterList& paramList)
  /******************************************************************************/
  {
      if(paramList.isSublist("Elastic Stiffness"))
      {
          auto tParams = paramList.sublist("Elastic Stiffness");
          if (tParams.isSublist("Youngs Modulus"))
          {
              this->setRank4VoigtFunctor("Elastic Stiffness", Plato::IsotropicStiffnessFunctor<SpatialDim>(tParams));
          }
          else
          if (tParams.isType<Plato::Scalar>("Youngs Modulus"))
          {
              this->setRank4VoigtConstant("Elastic Stiffness", Plato::IsotropicStiffnessConstant<SpatialDim>(tParams));
          }
          else
          {
              this->parseRank4Voigt("Elastic Stiffness", tParams);
          }
      }
  }

  /******************************************************************************/
  /*!
    \brief Factory for creating material models
  */
    template<int SpatialDim>
    class ThermoelasticModelFactory
  /******************************************************************************/
  {
    public:
      ThermoelasticModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
      Teuchos::RCP<Plato::MaterialModel<SpatialDim>> create(std::string aModelName);
    private:
      const Teuchos::ParameterList& mParamList;
  };

  /******************************************************************************/
  template<int SpatialDim>
  Teuchos::RCP<MaterialModel<SpatialDim>>
  ThermoelasticModelFactory<SpatialDim>::create(std::string aModelName)
  /******************************************************************************/
  {
      if (!mParamList.isSublist("Material Models"))
      {
          REPORT("'Material Models' list not found! Returning 'nullptr'");
          return Teuchos::RCP<Plato::MaterialModel<SpatialDim>>(nullptr);
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

          if( tModelParamList.isSublist("Thermoelastic") )
          {
            return Teuchos::rcp(new Plato::ThermoelasticMaterial<SpatialDim>(tModelParamList.sublist("Thermoelastic")));
          }
          else
          ANALYZE_THROWERR("Expected 'Thermoelastic' ParameterList");
      }

  }
} // namespace Plato
