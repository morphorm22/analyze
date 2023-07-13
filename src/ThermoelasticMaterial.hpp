#pragma once

#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"
#include "materials/MaterialModel.hpp"

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Base class for Thermoelastic material models
  */
    template<typename EvaluationType>
    class ThermoelasticMaterial : public MaterialModel<EvaluationType>
  /******************************************************************************/
  {
    private:
      using ElementType = typename EvaluationType::ElementType; // set local element type
      static constexpr int SpatialDim = ElementType::mNumSpatialDims;
    
    public:
      ThermoelasticMaterial(const Teuchos::ParameterList& paramList);

    private:
      void parseElasticStiffness(const Teuchos::ParameterList& paramList);
  };

  /******************************************************************************/
  template<typename EvaluationType>
  ThermoelasticMaterial<EvaluationType>::
  ThermoelasticMaterial(const Teuchos::ParameterList& paramList) : 
    MaterialModel<EvaluationType>(paramList)
  /******************************************************************************/
  {
      this->parseElasticStiffness(paramList);
      this->parseTensor("Thermal Expansivity", paramList);
      this->parseTensor("Thermal Conductivity", paramList);

      this->parseScalarConstant("Reference Temperature", paramList, 23.0);
      this->parseScalarConstant("Temperature Scaling", paramList, 1.0);
  }

  /******************************************************************************/
  template<typename EvaluationType>
  void ThermoelasticMaterial<EvaluationType>::
  parseElasticStiffness(const Teuchos::ParameterList& aParamList)
  /******************************************************************************/
  {
      if(aParamList.isSublist("Elastic Stiffness"))
      {
          auto tParams = aParamList.sublist("Elastic Stiffness");
          if (tParams.isSublist("Youngs Modulus"))
          {
              this->setRank4VoigtFunctor(
                "Elastic Stiffness", Plato::IsotropicStiffnessFunctor<SpatialDim>(tParams));
          }
          else
          if (tParams.isType<Plato::Scalar>("Youngs Modulus"))
          {
              this->setRank4VoigtConstant(
                "Elastic Stiffness", Plato::IsotropicStiffnessConstant<SpatialDim>(tParams));
          }
          else
          {
              this->parseRank4Voigt("Elastic Stiffness", aParamList);
          }
      }
  }

  /******************************************************************************/
  /*!
    \brief Factory for creating material models
  */
    template<typename EvaluationType>
    class ThermoelasticModelFactory
  /******************************************************************************/
  {
    private:
      using ElementType = typename EvaluationType::ElementType; // set local element type
      static constexpr int SpatialDim = ElementType::mNumSpatialDims;

    public:
      ThermoelasticModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
      Teuchos::RCP<Plato::MaterialModel<EvaluationType>> create(std::string aModelName);
    private:
      const Teuchos::ParameterList& mParamList;
  };

  /******************************************************************************/
  template<typename EvaluationType>
  Teuchos::RCP<MaterialModel<EvaluationType>>
  ThermoelasticModelFactory<EvaluationType>::create(std::string aModelName)
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

          if( tModelParamList.isSublist("Thermoelastic") )
          {
            return Teuchos::rcp(new Plato::ThermoelasticMaterial<EvaluationType>(
                                tModelParamList.sublist("Thermoelastic")));
          }
          else
          ANALYZE_THROWERR("Expected 'Thermoelastic' ParameterList");
      }

  }
} // namespace Plato
