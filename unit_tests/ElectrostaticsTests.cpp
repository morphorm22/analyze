/*
 * ElectrostaticsTests.cpp
 *
 *  Created on: May 10, 2023
 */

// c++ includes
#include <vector>
#include <unordered_map>

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// plato
#include "Tri3.hpp"
#include "Simp.hpp"
#include "Solutions.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ScalarGrad.hpp"
#include "WorksetBase.hpp"
#include "SpatialModel.hpp"
#include "MaterialModel.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"

#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace electrical 
{

enum struct property
{
    /// @brief Supported material property enums for electrical materials
    ELECTRICAL_CONDUCTIVITY=0, 
    OUT_OF_PLANE_THICKNESS=1, 
    MATERIAL_NAME=2, 
    ELECTRICAL_CONSTANT=3, 
    RELATIVE_STATIC_PERMITTIVITY=4,
    TO_ERSATZ_MATERIAL_EXPONENT=5,
    TO_MIN_ERSATZ_MATERIAL_VALUE=6
};
// enum struct property

struct PropEnum
{
private:
    std::unordered_map<std::string,Plato::electrical::property> s2e = {
        {"electrical conductivity"     ,Plato::electrical::property::ELECTRICAL_CONDUCTIVITY},
        {"out-of-plane thickness"      ,Plato::electrical::property::OUT_OF_PLANE_THICKNESS},
        {"material name"               ,Plato::electrical::property::MATERIAL_NAME},
        {"electrical constant"         ,Plato::electrical::property::ELECTRICAL_CONSTANT},
        {"relative static permittivity",Plato::electrical::property::RELATIVE_STATIC_PERMITTIVITY},
        {"penalty exponent"            ,Plato::electrical::property::TO_ERSATZ_MATERIAL_EXPONENT},
        {"minimum value"               ,Plato::electrical::property::TO_MIN_ERSATZ_MATERIAL_VALUE}    
    };

public:
    Plato::electrical::property 
    get(const std::string &aInput) 
    const
    {
        auto tLower = Plato::tolower(aInput);
        auto tItr = s2e.find(tLower);
        if( tItr == s2e.end() ){
            auto tMsg = this->getErrorMsg(tLower);
            ANALYZE_THROWERR(tMsg)
        }
        return tItr->second;
    }

private:
    std::string
    getErrorMsg(const std::string & aInProperty)
    const
    {
        auto tMsg = std::string("Did not find matching enum for input electrical property '") 
                + aInProperty + "'. Supported electrical property keywords are: ";
        for(const auto& tPair : s2e)
        {
            tMsg = tMsg + "'" + tPair.first + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }
};
// struct PropEnum

enum struct source
{
  /// @brief Supported enums for current density source evaluators
  SINGLE_DIODE=0, 
  TWO_PHASE_DARK_CURRENT_DENSITY=1, 
  TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY=2, 
};
// enum struct source 

struct SourceEnum
{
private:
    std::unordered_map<std::string,Plato::electrical::source> s2e = {
      {"single diode"                             ,Plato::electrical::source::SINGLE_DIODE},
      {"two phase dark current density"           ,Plato::electrical::source::TWO_PHASE_DARK_CURRENT_DENSITY},
      {"two phase light-generated current density",Plato::electrical::source::TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY}
    };

public:
    Plato::electrical::source 
    get(const std::string &aInput) 
    const
    {
        auto tLower = Plato::tolower(aInput);
        auto tItr = s2e.find(tLower);
        if( tItr == s2e.end() ){
            auto tMsg = this->getErrorMsg(tLower);
            ANALYZE_THROWERR(tMsg)
        }
        return tItr->second;
    }

private:
    std::string
    getErrorMsg(const std::string & aInProperty)
    const
    {
        auto tMsg = std::string("Did not find matching enum for input electrical source type '") 
                + aInProperty + "'. Supported electrical source keywords are: ";
        for(const auto& tPair : s2e)
        {
            tMsg = tMsg + "'" + tPair.first + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }
};
// struct SourceEnum

enum struct source_model
{
  /// @brief Supported enums for current density models
  DARK_CURRENT_DENSITY_QUADRATIC_FIT=0, 
  LIGHT_GENERATED_CURRENT_DENSITY_CONSTANT=1, 
};
// enum struct source_model 

struct SourceModelEnum
{
private:
  typedef std::unordered_map<Plato::electrical::source,std::unordered_map<std::string,Plato::electrical::source_model>>
    SourceModelMap;

  SourceModelMap mLinearS2E = 
    {
      { Plato::electrical::source::TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY, 
        {
          { "constant",Plato::electrical::source_model::LIGHT_GENERATED_CURRENT_DENSITY_CONSTANT } 
        }
      }
    };

  SourceModelMap mNonlinearS2E = 
    {
      { Plato::electrical::source::TWO_PHASE_DARK_CURRENT_DENSITY, 
        {
          { "quadratic",Plato::electrical::source_model::DARK_CURRENT_DENSITY_QUADRATIC_FIT } 
        }
      },
    };

public:
    Plato::electrical::source_model 
    get(
      const Plato::electrical::source & aSourceEnum,
      const std::string               & aSourceModel,
      const std::string               & aModelType
    ) const
    {
      Plato::electrical::source_model tOutput;
      auto tLowertType = Plato::tolower(aModelType);
      if( tLowertType.compare("linear") == 0 )
      {
        tOutput = this->model(aSourceEnum,aSourceModel,mLinearS2E);
      }
      else
      if( tLowertType.compare("nonlinear") == 0 )
      {
        tOutput = this->model(aSourceEnum,aSourceModel,mNonlinearS2E);
      }
      else
      {
        auto tMsg = std::string("Source models of type ('") + tLowertType + "') is not supported, " 
          + "supported source model types are: 'linear' and 'nonlinear'";
        ANALYZE_THROWERR(tMsg) 
      }
      return tOutput;
    }

private:
    Plato::electrical::source_model 
    model(
      const Plato::electrical::source & aSourceEnum,
      const std::string               & aSourceModel,
      const SourceModelMap            & aMap
    ) const
    {
      auto tItrOne = aMap.find(aSourceEnum);
      if( tItrOne == aMap.end() )
      {
        auto tMsg = std::string("Requested source term does not support a current density model");
        ANALYZE_THROWERR(tMsg)
      }
      auto tLower = Plato::tolower(aSourceModel);
      auto tItrTwo = tItrOne->second.find(tLower);
      if( tItrTwo == tItrOne->second.end() )
      {
        auto tMsg = this->getErrorMsg(aSourceEnum,tLower,aMap);
        ANALYZE_THROWERR(tMsg)
      }
      return tItrTwo->second;
    }

    std::string
    getErrorMsg(
      const Plato::electrical::source & aSourceEnum,
      const std::string               & aSourceModel,
      const SourceModelMap            & aMap
    ) const
    {
        auto tMsg = std::string("Did not find matching enum for input electrical source model '") 
                + aSourceModel + "'. Supported electrical source model keywords for requested source term are: ";
        for(const auto& tPair : aMap.find(aSourceEnum)->second)
        {
            tMsg = tMsg + "'" + tPair.first + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }
};
// struct SourceEnum

};
// namespace electrical 

/******************************************************************************/
/*!
 \brief Base class for linear electrical conductivity material model
 */
template<typename EvaluationType>
class MaterialElectricalConductivity : public MaterialModel<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType; // set local element type
    static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
    
    Plato::electrical::PropEnum mS2E; /*!< map string to supported enum */
    std::unordered_map<Plato::electrical::property,std::vector<std::string>> mProperties;

public:
    MaterialElectricalConductivity(
        const std::string            & aMaterialName,
        const Teuchos::ParameterList & aParamList
    )
    {
        this->name(aMaterialName);
        this->parseScalar("Electrical Conductivity", aParamList);
        auto tElectricConductivity = this->getScalarConstant("Electrical Conductivity");
        this->setTensorConstant("material tensor",Plato::TensorConstant<mNumSpatialDims>(tElectricConductivity));
        mProperties[mS2E.get("Electrical Conductivity")].push_back( std::to_string(tElectricConductivity) );
    }
    ~MaterialElectricalConductivity(){}

    std::vector<std::string> 
    property(const std::string & aPropertyID)
    const override
    {
        auto tEnum = mS2E.get(aPropertyID);
        auto tItr = mProperties.find(tEnum);
        if( tItr == mProperties.end() ){
            return {};
        }
        return tItr->second;
    }
};

/******************************************************************************/
/*!
 \brief Base class for linear electrical conductivity material model
 */
template<typename EvaluationType>
class MaterialElectricalConductivityTwoPhaseAlloy : public MaterialModel<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType; // set local element type
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;

    // define ad types
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    
    Plato::Scalar mPenaltyExponent = 3.0;
    Plato::Scalar mMinErsatzMaterialValue = 0.0;

    std::vector<std::string>   mMaterialNames;
    std::vector<Plato::Scalar> mConductivities;
    std::vector<Plato::Scalar> mOutofPlaneThickness;

    Plato::electrical::PropEnum mS2E; /*!< map string to supported enum */
    std::unordered_map<Plato::electrical::property,std::vector<std::string>> mProperties;

public:
    MaterialElectricalConductivityTwoPhaseAlloy(
        const std::string            & aMaterialName, 
              Teuchos::ParameterList & aParamList
    )
    {
        this->name(aMaterialName);
        this->initialize(aParamList);
    }
    ~MaterialElectricalConductivityTwoPhaseAlloy(){}

    void 
    computeMaterialTensor(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray4DT<ResultScalarType>      & aResult
    ) 
    override
    { 
        // get material tensor for each phase
        const Plato::TensorConstant<mNumSpatialDims> tTensorOne = this->getTensorConstant(mMaterialNames.front());
        const Plato::TensorConstant<mNumSpatialDims> tTensorTwo = this->getTensorConstant(mMaterialNames.back());

        // create material penalty model
        Plato::MSIMP tSIMP(mPenaltyExponent,mMinErsatzMaterialValue);

        // evaluate residual    
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumCells   = aSpatialDomain.numCells();
        auto tNumPoints  = tCubWeights.size();     

        Kokkos::parallel_for("evaluate material tensor for 2-phase alloy", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ControlScalarType tDensity = 
              Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControl, tBasisValues);
            ControlScalarType tMaterialPenalty = tSIMP(tDensity);
            for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
                for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
                    aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) = tTensorTwo(tDimI,tDimJ) + 
                      ( ( tTensorOne(tDimI,tDimJ) - tTensorTwo(tDimI,tDimJ) ) * tMaterialPenalty );
                }
            }
        });
    }

    std::vector<std::string> 
    property(const std::string & aPropertyID)
    const override
    {
        auto tEnum = mS2E.get(aPropertyID);
        auto tItr = mProperties.find(tEnum);
        if( tItr == mProperties.end() ){
            return {};
        }
        return tItr->second;
    }

private:
    void initialize(
        Teuchos::ParameterList & aParamList
    )
    {
        this->parsePhysicalProperties(aParamList);
        this->parseOutofPlaneThickness(aParamList);
        this->parseMaterialNames(aParamList);
        this->parsePenaltyModel(aParamList);
        this->setTensors();
    }

    void parsePhysicalProperties(
        Teuchos::ParameterList & aParamList
    )
    {
        bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Electrical Conductivity");
        if (tIsArray)
        {
            // parse inputs
            Teuchos::Array<Plato::Scalar> tConductivities = 
              aParamList.get<Teuchos::Array<Plato::Scalar>>("Electrical Conductivity");
            if(tConductivities.size() != 2){
              auto tMaterialName = this->name();
              auto tMsg = std::string("Size of electrical conductivity array must equal two. ") 
                + "Check electrical conductivity inputs in material block with name '" + tMaterialName
                + "'. Material tensor cannnot be computed.";
              ANALYZE_THROWERR(tMsg)
            }
            // create mirror 
            for(size_t tIndex = 0; tIndex < tConductivities.size(); tIndex++)
            {
              mProperties[mS2E.get("Electrical Conductivity")].push_back( std::to_string(tConductivities[tIndex]) );
              mConductivities.push_back(tConductivities[tIndex]);
            }
        }
        else
        {
            auto tMaterialName = this->name();
            auto tMsg = std::string("Array of electrical conductivities is not defined in material block with name '") 
              + tMaterialName + "'. Material tensor for a two-phase alloy cannnot be computed.";
            ANALYZE_THROWERR(tMsg)
        }
    }
    
    void 
    parseMaterialNames(
        Teuchos::ParameterList & aParamList
    )
    {
        bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Material Name");
        if (tIsArray)
        {
            // parse inputs
            Teuchos::Array<std::string> tMaterialNames = 
              aParamList.get<Teuchos::Array<std::string>>("Material Name");
            // create mirror 
            for(size_t tIndex = 0; tIndex < mMaterialNames.size(); tIndex++){
                mMaterialNames.push_back(tMaterialNames[tIndex]);
            }
            if( mConductivities.size() > mMaterialNames.size() ){
                // assume default values for missing names
                for(Plato::OrdinalType tIndex = mMaterialNames.size() - 1u; tIndex < mConductivities.size(); tIndex++){
                    auto tName = std::string("material ") + std::to_string(tIndex);
                    mProperties[mS2E.get("Material Name")].push_back(tName);
                    mMaterialNames.push_back(tName);
                }
            }
        }
        else
        {
            // assuming default names
            for(Plato::OrdinalType tIndex = 0; tIndex < mConductivities.size(); tIndex++)
            {
                auto tName = std::string("material ") + std::to_string(tIndex);
                mProperties[mS2E.get("Material Name")].push_back(tName);
                mMaterialNames.push_back(tName);
            }
        }
    }

    void 
    parseOutofPlaneThickness(
        Teuchos::ParameterList &aParamList
    )
    {
        if(mNumSpatialDims >= 3){
            return;
        }

        bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Out-of-Plane Thickness");
        if (tIsArray)
        {
            // parse inputs
            Teuchos::Array<Plato::Scalar> tOutofPlaneThickness = 
              aParamList.get<Teuchos::Array<Plato::Scalar>>("Out-of-Plane Thickness");
            if(tOutofPlaneThickness.size() != 2){
              auto tMaterialName = this->name();
              auto tMsg = std::string("Size of out-of-plane thickness array must equal two. ") 
                + "Check out-of-plane thickness inputs in material block with name '" + tMaterialName
                + "'. Material tensor cannnot be computed.";
              ANALYZE_THROWERR(tMsg)
            }
            // create mirror 
            mOutofPlaneThickness.clear();
            for(size_t tIndex = 0; tIndex < tOutofPlaneThickness.size(); tIndex++)
            {
              mProperties[mS2E.get("Out-of-Plane Thickness")].push_back(std::to_string(tOutofPlaneThickness[tIndex]));
              mOutofPlaneThickness.push_back(tOutofPlaneThickness[tIndex]);
            }
        }
        else
        {
            auto tMsg = std::string("Requested an electrical conductivity material constitutive model for ") 
              + "a two-phase alloy modeled in two dimensions but array of out-of-plane thicknesses is not defined. "
              + "Physics of interest cannot be accurately simulated.";
            ANALYZE_THROWERR(tMsg)
        }
    }

    void 
    parsePenaltyModel(
        Teuchos::ParameterList & aParamList
    )
    {
        mPenaltyExponent = aParamList.get<Plato::Scalar>("Penalty Exponent", 3.0);
        mProperties[mS2E.get("Penalty Exponent")].push_back( std::to_string(mPenaltyExponent) );
        mMinErsatzMaterialValue = aParamList.get<Plato::Scalar>("Minimum Value", 0.0);
        mProperties[mS2E.get("Minimum Value")].push_back( std::to_string(mMinErsatzMaterialValue) );
    }

    void 
    setTensors()
    {
        for(const auto& tConductivity : mConductivities){
            Plato::OrdinalType tIndex = &tConductivity - &mConductivities[0];
            this->setTensorConstant(mMaterialNames[tIndex],Plato::TensorConstant<mNumSpatialDims>(tConductivity));
        }
    }
};

/******************************************************************************/
/*!
 \brief Base class for linear dielectric material model
 */
template<typename EvaluationType>
class MaterialDielectric : public MaterialModel<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType; // set local element type
    static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;

    Plato::electrical::PropEnum mS2E; /*!< map string to supported enum */
    std::unordered_map<Plato::electrical::property,std::vector<std::string>> mProperties;

public:
    MaterialDielectric(
        const std::string            & aMaterialName, 
              Teuchos::ParameterList & aParamList
    )
    {
        this->name(aMaterialName);
        this->initialize(aParamList);
    }
    ~MaterialDielectric(){}

    std::vector<std::string> 
    property(const std::string & aPropertyID)
    const override
    {
        auto tEnum = mS2E.get(aPropertyID);
        auto tItr = mProperties.find(tEnum);
        if( tItr == mProperties.end() ){
            return {};
        }
        return tItr->second;
    }

private:
    void 
    initialize(
        Teuchos::ParameterList & aParamList
    )
    {
        this->parseScalar("Electrical Constant", aParamList);
        this->parseScalar("Relative Static Permittivity", aParamList);
        auto tElectricConstant = this->getScalarConstant("Electrical Constant");
        auto tRelativeStaticPermittivity = this->getScalarConstant("Relative Static Permittivity");
        auto tValue = tElectricConstant * tRelativeStaticPermittivity;
        this->setTensorConstant("material tensor",Plato::TensorConstant<mNumSpatialDims>(tValue));

        mProperties[mS2E.get("Electrical Constant")].push_back(std::to_string(tElectricConstant));
        mProperties[mS2E.get("Relative Static Permittivity")].push_back(std::to_string(tRelativeStaticPermittivity));
    }
};

/******************************************************************************/
/*!
 \brief Factory for creating electrical material constitutive models
 */
template<typename EvaluationType>
class FactoryElectricalMaterial
/******************************************************************************/
{
public:
    FactoryElectricalMaterial(
        Teuchos::ParameterList& aParamList
    ) :
      mParamList(aParamList)
    {
    }

    std::shared_ptr<MaterialModel<EvaluationType>> create(std::string aModelName)
    {
        if (!mParamList.isSublist("Material Models"))
        {
            ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
        }
        else
        {
            auto tModelsParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");
            if (!tModelsParamList.isSublist(aModelName))
            {
                auto tMsg = std::string("Requested a material model with name ('") + aModelName 
                            + "') that is not defined in the input deck";
                ANALYZE_THROWERR(tMsg);
            }
    
            auto tModelParamList = tModelsParamList.sublist(aModelName);
            if(tModelParamList.isSublist("Two Phase Electrical Conductivity"))
            {
                auto tMaterial = std::make_shared<Plato::MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>>
                                 (aModelName, tModelParamList.sublist("Two Phase Electrical Conductivity"));
                tMaterial->model("Two Phase Electrical Conductivity");
                return tMaterial;
            }
            else
            if(tModelParamList.isSublist("Electrical Conductivity"))
            {
                auto tMaterial = std::make_shared<Plato::MaterialElectricalConductivity<EvaluationType>>
                                 (aModelName, tModelParamList.sublist("Electrical Conductivity"));
                tMaterial->model("Electrical Conductivity");
                return tMaterial;
            }
            else 
            if(tModelParamList.isSublist("Dielectric"))
            {
                auto tMaterial = std::make_shared<Plato::MaterialDielectric<EvaluationType>>
                                (aModelName, tModelParamList.sublist("Dielectric"));
                tMaterial->model("Dielectric");
                return tMaterial;
            }
            else
            {
                auto tErrMsg = this->getErrorMsg();
                ANALYZE_THROWERR(tErrMsg);
            }
        }
    }

private:
    std::string
    getErrorMsg()
    const
    {
        std::string tMsg = std::string("ERROR: Requested material constitutive model is not supported. ")
            + "Supported material constitutive models for a steady state current analysis are: ";
        for(const auto& tElement : mSupportedMaterials)
        {
            tMsg = tMsg + "'" + tElement + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }

private:
    const Teuchos::ParameterList& mParamList;
    
    std::vector<std::string> mSupportedMaterials = 
      {"Electrical Conductivity", "Dielectric", "Two Phase Electrical Conductivity"};
};

template<typename EvaluationType, 
         typename OutputScalarType>
class CurrentDensityModel
{
private:
    using ElementType = typename EvaluationType::ElementType;
    using StateScalarType   = typename EvaluationType::StateScalarType;

public:
    CurrentDensityModel(){}
    virtual ~CurrentDensityModel(){}

    KOKKOS_INLINE_FUNCTION
    virtual 
    OutputScalarType 
    evaluate(
        const StateScalarType & aCellElectricPotential
    ) const = 0;
};

template<typename EvaluationType, 
         typename OutputScalarType = typename EvaluationType::StateScalarType>
class DarkCurrentDensityQuadraticFit : 
    public Plato::CurrentDensityModel<EvaluationType,OutputScalarType>
{
private:
    using ElementType = typename EvaluationType::ElementType;
    using StateScalarType   = typename EvaluationType::StateScalarType;

public:
    Plato::Scalar mCoefA  = 0.;
    Plato::Scalar mCoefB  = 1.27e-6;
    Plato::Scalar mCoefC  = 25.94253;
    Plato::Scalar mCoefM1 = 0.38886;
    Plato::Scalar mCoefB1 = 0.;
    Plato::Scalar mCoefM2 = 30.;
    Plato::Scalar mCoefB2 = 6.520373;
    Plato::Scalar mPerformanceLimit = -0.22;

    std::string mCurrentDensityName = "";

public:
    DarkCurrentDensityQuadraticFit(
      const std::string            & aCurrentDensityName,
      const Teuchos::ParameterList & aParamList
    ) : 
      mCurrentDensityName(aCurrentDensityName)
    {
        this->initialize(aParamList);
    }
    virtual ~DarkCurrentDensityQuadraticFit(){}
    
    KOKKOS_INLINE_FUNCTION
    OutputScalarType 
    evaluate(
        const StateScalarType & aCellElectricPotential
    ) const
    {
        OutputScalarType tDarkCurrentDensity = 0.0;
        if( aCellElectricPotential > 0.0 )
          { tDarkCurrentDensity = mCoefA + mCoefB * exp(mCoefC * aCellElectricPotential); }
        else 
        if( (mPerformanceLimit < aCellElectricPotential) && (aCellElectricPotential < 0.0) )
          { tDarkCurrentDensity = mCoefM1 * aCellElectricPotential + mCoefB1; }
        else 
        if( aCellElectricPotential < mPerformanceLimit )
          { tDarkCurrentDensity = mCoefM2 * aCellElectricPotential + mCoefB2; }
        return tDarkCurrentDensity;
    }

private:
    void 
    initialize(
      const Teuchos::ParameterList & aParamList
    )
    {
        if( !aParamList.isSublist("Source Terms") ){
          auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
          ANALYZE_THROWERR(tMsg)
        }
        auto tSourceTermsSublist = aParamList.sublist("Source Terms");

        if( !tSourceTermsSublist.isSublist(mCurrentDensityName) ){
          auto tMsg = std::string("Parameter is not valid. Argument ('") + mCurrentDensityName 
            + "') is not a parameter list";
          ANALYZE_THROWERR(tMsg)
        }
        auto tCurrentDensitySublist = tSourceTermsSublist.sublist(mCurrentDensityName);
        this->parseParameters(tCurrentDensitySublist);
    }
    void 
    parseParameters(
      Teuchos::ParameterList & aParamList
    )
    {
        mCoefA  = aParamList.get<Plato::Scalar>("a",0.);
        mCoefB  = aParamList.get<Plato::Scalar>("b",1.27e-6);
        mCoefC  = aParamList.get<Plato::Scalar>("c",25.94253);
        mCoefM1 = aParamList.get<Plato::Scalar>("m1",0.38886);
        mCoefB1 = aParamList.get<Plato::Scalar>("b1",0.);
        mCoefM2 = aParamList.get<Plato::Scalar>("m2",30.);
        mCoefB2 = aParamList.get<Plato::Scalar>("b2",6.520373);
        mPerformanceLimit = aParamList.get<Plato::Scalar>("limit",-0.22);
    }
};

template<typename EvaluationType, 
         typename OutputScalarType = Plato::Scalar>
class LightGeneratedCurrentDensityConstant : 
    public Plato::CurrentDensityModel<EvaluationType,OutputScalarType>
{
private:
    using ElementType = typename EvaluationType::ElementType;
    using StateScalarType = typename EvaluationType::StateScalarType;

public:
    std::string mCurrentDensityName = ""; /*!< input light-generated current density parameter list name */
    Plato::Scalar mGenerationRate = -0.40914; /*!< generation rate */
    Plato::Scalar mIlluminationPower = 1000.0; /*!< solar illumination power */

public:
    LightGeneratedCurrentDensityConstant(
      const std::string            & aCurrentDensityName,
      const Teuchos::ParameterList & aParamList
    ) : 
      mCurrentDensityName(aCurrentDensityName)
    {
      this->initialize(aParamList);
    }
    virtual ~LightGeneratedCurrentDensityConstant(){}

    KOKKOS_INLINE_FUNCTION
    OutputScalarType 
    evaluate(
        const StateScalarType & aCellElectricPotential
    ) const
    {
      Plato::Scalar tOutput = mGenerationRate * mIlluminationPower;
      return ( tOutput );
    }

private:
    void 
    initialize(
      const Teuchos::ParameterList &aParamList
    )
    {
        if( !aParamList.isSublist("Source Terms") ){
          auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
          ANALYZE_THROWERR(tMsg)
        }
        auto tSourceTermsSublist = aParamList.sublist("Source Terms");

        if( !tSourceTermsSublist.isSublist(mCurrentDensityName) ){
          auto tMsg = std::string("Parameter is not valid. Argument ('") + mCurrentDensityName 
            + "') is not a parameter list";
          ANALYZE_THROWERR(tMsg)
        }
        auto tCurrentDensitySublist = tSourceTermsSublist.sublist(mCurrentDensityName);
        mGenerationRate = tCurrentDensitySublist.get<Plato::Scalar>("Generation Rate",-0.40914);
        mIlluminationPower = tCurrentDensitySublist.get<Plato::Scalar>("Illumination Power",1000.);
    }
};

namespace FactoryCurrentDensityModel
{
// TODO: FINISH FACTORY
}


template<typename EvaluationType>
class CurrentDensityEvaluator
{
private:
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

public:
    virtual 
    void 
    evaluate(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
        const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
              Plato::Scalar                                  aScale
    ) 
    const = 0;
};

template<typename EvaluationType>
class LightCurrentDensityTwoPhaseAlloy : 
  public Plato::CurrentDensityEvaluator<EvaluationType>
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::string mMaterialName = "";
    std::string mCurrentDensityName  = "";
    std::string mCurrentDensityType  = "Linear";
    std::string mCurrentDensityModel = "Constant";

    Plato::Scalar mPenaltyExponent = 3.0; /*!< penalty exponent for material penalty model */
    Plato::Scalar mMinErsatzMaterialValue = 0.0; /*!< minimum value for the ersatz material density */
    std::vector<Plato::Scalar> mOutofPlaneThickness; /*!< list of out-of-plane material thickness */

    const Teuchos::ParameterList& mParamList;

public:
    LightCurrentDensityTwoPhaseAlloy(
      const std::string            & aMaterialName,
      const std::string            & aCurrentDensityName,
            Teuchos::ParameterList & aParamList
    ) : 
      mMaterialName(aMaterialName),
      mCurrentDensityName(aCurrentDensityName),
      mParamList(aParamList)
    {
        this->initialize(aParamList);
    }
    ~LightCurrentDensityTwoPhaseAlloy(){}

    void 
    evaluate(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
        const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
              Plato::Scalar                                  aScale
    ) 
    const
    {
        // integration rule
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // interpolate nodal values to integration points
        Plato::LightGeneratedCurrentDensityConstant<EvaluationType> 
          tLightGeneratedCurrentDensityModel(mCurrentDensityName,mParamList);
        Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;

        // out-of-plane thicknesses
        Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
        Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();

        // evaluate light-generated current density
        Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
        Kokkos::parallel_for("light-generated current density", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            // material interpolation
            ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
            ControlScalarType tMaterialFraction = static_cast<Plato::Scalar>(1.0) - tDensity;
            ControlScalarType tMaterialPenalty = pow(tMaterialFraction, mPenaltyExponent);

            // out-of-plane thickness interpolation
            ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
            ControlScalarType tThicknessInterpolation = tThicknessTwo + 
              ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );

            // evaluate light-generated current density
            StateScalarType tCellElectricPotential = tInterpolateFromNodal(iCellOrdinal,tBasisValues,aState);
            Plato::Scalar tLightGenCurrentDensity = 
              tLightGeneratedCurrentDensityModel.evaluate(tCellElectricPotential);

            auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
            {
                ResultScalarType tCellResult = ( tBasisValues(tFieldOrdinal) * 
                (tMaterialPenalty * tLightGenCurrentDensity) * tWeight ) / tThicknessInterpolation; 
                Kokkos::atomic_add( &aResult(iCellOrdinal,tFieldOrdinal*mNumDofsPerNode),tCellResult );
            }
        });
    }

private:
    void 
    initialize(
        Teuchos::ParameterList &aParamList
    )
    {
        if( !aParamList.isSublist("Source Terms") ){
          auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
          ANALYZE_THROWERR(tMsg)
        }
        auto tSourceTermsSublist = aParamList.sublist("Source Terms");

        if( !tSourceTermsSublist.isSublist(mCurrentDensityName) ){
          auto tMsg = std::string("Parameter is not valid. Argument ('") + mCurrentDensityName 
            + "') is not a parameter list";
          ANALYZE_THROWERR(tMsg)
        }
        auto tCurrentDensitySublist = tSourceTermsSublist.sublist(mCurrentDensityName);
        mPenaltyExponent = tCurrentDensitySublist.get<Plato::Scalar>("Penalty Exponent", 3.0);
        mMinErsatzMaterialValue = tCurrentDensitySublist.get<Plato::Scalar>("Minimum Value", 0.0);

        // set current density model
        mCurrentDensityType  = tCurrentDensitySublist.get<std::string>("Type","Linear");
        mCurrentDensityModel = tCurrentDensitySublist.get<std::string>("Model","Constant");

        // set out-of-plane thickness array
        Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
        auto tMaterialModel = tMaterialFactory.create(mMaterialName);
        this->setOutofPlaneThickness(tMaterialModel.operator*());
    }

    void 
    setOutofPlaneThickness(
        Plato::MaterialModel<EvaluationType> & aMaterialModel
    )
    {
        std::vector<std::string> tThickness = aMaterialModel.property("out-of-plane thickness");
        if ( tThickness.empty() )
        {
            auto tMsg = std::string("Array of out-of-plane thicknesses is empty. ") 
              + "Light-generated current density cannnot be computed.";
            ANALYZE_THROWERR(tMsg)
        }
        else
        {
            mOutofPlaneThickness.clear();
            for(size_t tIndex = 0; tIndex < tThickness.size(); tIndex++)
            {
              Plato::Scalar tMyThickness = std::stod(tThickness[tIndex]);
              mOutofPlaneThickness.push_back(tMyThickness);
            }
        }
    }
};

template<typename EvaluationType>
class DarkCurrentDensityTwoPhaseAlloy : 
  public Plato::CurrentDensityEvaluator<EvaluationType>
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::string mMaterialName = "";
    std::string mCurrentDensityName  = "";
    std::string mCurrentDensityType  = "Linear";
    std::string mCurrentDensityModel = "Quadratic";

    Plato::Scalar mPenaltyExponent = 3.0;
    Plato::Scalar mMinErsatzMaterialValue = 0.0;

    const Teuchos::ParameterList& mParamList;
    std::vector<Plato::Scalar> mOutofPlaneThickness;

public:
    DarkCurrentDensityTwoPhaseAlloy(
      const std::string            & aMaterialName,
      const std::string            & aCurrentDensityName,
            Teuchos::ParameterList & aParamList
    ) : 
      mMaterialName(aMaterialName),
      mCurrentDensityName(aCurrentDensityName),
      mParamList(aParamList)
    {
        this->initialize(aParamList);
    }
    ~DarkCurrentDensityTwoPhaseAlloy(){}

    void
    evaluate(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
        const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
              Plato::Scalar                                  aScale
    ) const
    {
        // integration rule
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // out-of-plane thicknesses
        Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
        Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();

        // create functors: 1) compute dark current density and 2) interpolate nodal values to integration points
        Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;
        Plato::DarkCurrentDensityQuadraticFit<EvaluationType> 
          tDarkCurrentDensityModel(mCurrentDensityName,mParamList);

        // evaluate dark current density
        Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
        Kokkos::parallel_for("dark current density", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            // get basis functions and weights for this integration point
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            // out-of-plane thickness interpolation
            ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
            ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
            ControlScalarType tThicknessInterpolation = tThicknessTwo + 
              ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );

            // compute dark current density
            StateScalarType tCellElectricPotential = tInterpolateFromNodal(iCellOrdinal,tBasisValues,aState);
            StateScalarType tDarkCurrentDensity = tDarkCurrentDensityModel.evaluate(tCellElectricPotential);

            auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
            {
                ResultScalarType tCellResult = 
                    ( tBasisValues(tFieldOrdinal) * tDarkCurrentDensity * tWeight ) / tThicknessInterpolation;
                Kokkos::atomic_add( &aResult(iCellOrdinal,tFieldOrdinal*mNumDofsPerNode), tCellResult );
            }
        });
    }

private:
    void 
    initialize(
        Teuchos::ParameterList &aParamList
    )
    {
        if( !aParamList.isSublist("Source Terms") ){
          auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
          ANALYZE_THROWERR(tMsg)
        }
        auto tSourceTermsSublist = aParamList.sublist("Source Terms");

        if( !tSourceTermsSublist.isSublist(mCurrentDensityName) ){
          auto tMsg = std::string("Parameter is not valid. Argument ('") + mCurrentDensityName 
            + "') is not a parameter list";
          ANALYZE_THROWERR(tMsg)
        }
        Teuchos::ParameterList& tSublist = aParamList.sublist(mCurrentDensityName);
        mPenaltyExponent = tSublist.get<Plato::Scalar>("Penalty Exponent", 3.0);
        mMinErsatzMaterialValue = tSublist.get<Plato::Scalar>("Minimum Value", 0.0);

        // set current density model
        mCurrentDensityType  = tSublist.get<std::string>("Type","Nonlinear");
        mCurrentDensityModel = tSublist.get<std::string>("Model","Quadratic");

        // set out-of-plane thickness array
        Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
        auto tMaterialModel = tMaterialFactory.create(mMaterialName);
        this->setOutofPlaneThickness(tMaterialModel.operator*());
    }
    
    void 
    setOutofPlaneThickness(
        Plato::MaterialModel<EvaluationType> & aMaterialModel
    )
    {
        std::vector<std::string> tThickness = aMaterialModel.property("out-of-plane thickness");
        if ( tThickness.empty() )
        {
            auto tMsg = std::string("Array of out-of-plane thicknesses is empty. ") 
              + "Dark current density cannnot be computed.";
            ANALYZE_THROWERR(tMsg)
        }
        else
        {
            mOutofPlaneThickness.clear();
            for(size_t tIndex = 0; tIndex < tThickness.size(); tIndex++)
            {
              Plato::Scalar tMyThickness = std::stod(tThickness[tIndex]);
              mOutofPlaneThickness.push_back(tMyThickness);
            }
        }
    }
};

namespace FactoryCurrentDensityEvaluator
{
  template <typename EvaluationType>
  inline std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> 
  create(
    const std::string            & aMaterialName,
    const std::string            & aFunctionName,
          Teuchos::ParameterList & aParamList
  )
  {
    if( !aParamList.isSublist("Source Terms") )
    {
      auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
      ANALYZE_THROWERR(tMsg)
    }
    auto tSourceTermsParamList = aParamList.sublist("Source Terms");
    if( !tSourceTermsParamList.isSublist(aFunctionName) )
    {
      auto tMsg = std::string("Parameter is not valid. Argument ('") + aFunctionName + "') is not a parameter list";
      ANALYZE_THROWERR(tMsg)
    }
    auto tCurrentDensityEvaluatorParamList = tSourceTermsParamList.sublist(aFunctionName);
    if( !tCurrentDensityEvaluatorParamList.isParameter("Function") )
    {
      auto tMsg = std::string("Parameter ('Function') is not defined in parameter list ('") 
        + aFunctionName + "'), current density evaluator cannot be determined";
      ANALYZE_THROWERR(tMsg)
    }
    Plato::electrical::SourceEnum tS2E;
    auto tType = tCurrentDensityEvaluatorParamList.get<std::string>("Function");
    auto tLowerType = Plato::tolower(tType);
    auto tSupportedSourceEnum = tS2E.get(tLowerType);
    switch (tSupportedSourceEnum)
    {
      case Plato::electrical::source::TWO_PHASE_DARK_CURRENT_DENSITY:
        return std::make_shared<Plato::DarkCurrentDensityTwoPhaseAlloy<EvaluationType>>(
          aMaterialName,aFunctionName,aParamList);
        break;
      case Plato::electrical::source::TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY:
        return std::make_shared<Plato::LightCurrentDensityTwoPhaseAlloy<EvaluationType>>(
          aMaterialName,aFunctionName,aParamList);
        break;
      case Plato::electrical::source::SINGLE_DIODE:
      default:
        return nullptr;
        break;
    }
  }
}
// namespace FactoryCurrentDensityEvaluator

// TODO: create factory to allocate volume forces. it will require refactoring the interfaces 
// to take workset metadata; e.g., similar to how it is done in the Ifem unit test. 
template<typename EvaluationType>
class SingleDiodeTwoPhaseAlloy
{
private:
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::string mMaterialName = "";
    std::vector<std::string>   mFunctions;
    std::vector<Plato::Scalar> mFunctionWeights;
    std::vector<std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>>> mCurrentDensityEvaluators;

public:
    SingleDiodeTwoPhaseAlloy(
      const std::string            & aMaterialName,
            Teuchos::ParameterList & aParamList
    ) : 
      mMaterialName(aMaterialName)
    {
      this->initialize(aParamList);
    }
    ~SingleDiodeTwoPhaseAlloy(){}

    void 
    evaluate(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
        const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
              Plato::Scalar                                  aScale
    ) 
    const
    {
      for(auto& tFunction : mCurrentDensityEvaluators)
      {
        Plato::OrdinalType tFunctionIndex = &tFunction - &mCurrentDensityEvaluators[0];
        Plato::Scalar tScalarMultiplier = mFunctionWeights[tFunctionIndex] * aScale;
        tFunction->evaluate(aSpatialDomain,aState,aControl,aConfig,aResult,tScalarMultiplier);
      }
    }

private:
    void initialize(
      Teuchos::ParameterList & aParamList
    )
    {
      if( !aParamList.isSublist("Source Terms") ){
        auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
        ANALYZE_THROWERR(tMsg)
      }
      auto tSourceTermsParamList = aParamList.sublist("Source Terms");
      if( !tSourceTermsParamList.isSublist("Single Diode") ){
        auto tMsg = std::string("Parameter is not valid. Argument ('Single Diode') is not a parameter list");
        ANALYZE_THROWERR(tMsg)
      }
      auto tSingleDiodeParamList = tSourceTermsParamList.sublist("Single Diode");
      this->parseFunctions(tSingleDiodeParamList);
      this->parseWeights(tSingleDiodeParamList);
      this->createCurrentDensityEvaluators(aParamList);
    }

    void parseFunctions(
      Teuchos::ParameterList & aParamList
    )
    {
      bool tIsArray = aParamList.isType<Teuchos::Array<std::string>>("Functions");
      if( !tIsArray)
      {
        auto tMsg = std::string("Argument ('Functions') is not defined in ('Single Diode') parameter list");
        ANALYZE_THROWERR(tMsg)
      }
      Teuchos::Array<std::string> tFunctions = aParamList.get<Teuchos::Array<std::string>>("Functions");
      for( Plato::OrdinalType tIndex = 0; tIndex < tFunctions.size(); tIndex++ )
        { mFunctions.push_back(tFunctions[tIndex]); }
    }

    void parseWeights(
      Teuchos::ParameterList & aParamList      
    )
    {
      bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Weights");
      if( !tIsArray)
      {
        if(mFunctions.size() <= 0)
        {
          auto tMsg = std::string("Argument ('Functions') has not been parsed, ") 
            + "default number of weights cannot be determined";
          ANALYZE_THROWERR(tMsg)
        }
        for( Plato::OrdinalType tIndex = 0; tIndex < mFunctions.size(); tIndex++ )
          { mFunctionWeights.push_back(1.0); }
      }
      else
      {
        Teuchos::Array<Plato::Scalar> tWeights = aParamList.get<Teuchos::Array<Plato::Scalar>>("Weights");
        for( Plato::OrdinalType tIndex = 0; tIndex < tWeights.size(); tIndex++ )
          { mFunctionWeights.push_back(tWeights[tIndex]); }
      }
    }

    void createCurrentDensityEvaluators(
      Teuchos::ParameterList & aParamList  
    )
    {
      for(auto& tFunctionName : mFunctions)
      {
        std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> tEvaluator = 
          Plato::FactoryCurrentDensityEvaluator::create<EvaluationType>(mMaterialName,tFunctionName,aParamList);
        mCurrentDensityEvaluators.push_back(tEvaluator);
      }
    }
};

namespace Elliptic
{

template<typename EvaluationType>
class PowerSurfaceDensity : public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr int mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;
 
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
    
    Plato::Scalar mPenaltyExponent = 3.0;

    std::vector<Plato::Scalar> mOutofPlaneThickness;

public:
    PowerSurfaceDensity(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aParamList, aFuncName)
    {
        Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
        auto tMaterialModel = tMaterialFactory.create(mSpatialDomain.getMaterialName());
        this->setOutofPlaneThickness(tMaterialModel.operator*());
    }
    ~PowerSurfaceDensity(){}

    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        // integration rule
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // out-of-plane thicknesses
        Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
        Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();

        // create functors: 1) compute dark current density and 2) interpolate nodal values to integration points
        Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;
        Plato::DarkCurrentDensityQuadraticFit<EvaluationType,StateScalarType> tDarkCurrentDensityModel;
        Plato::LightGeneratedCurrentDensityConstant<EvaluationType,Plato::Scalar> tLightCurrentDensityModel;

        // evaluate dark current density
        Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
        Kokkos::parallel_for("dark current density", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            // get basis functions and weights for this integration point
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            // material interpolation
            ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
            ControlScalarType tMaterialFraction = static_cast<Plato::Scalar>(1.0) - tDensity;
            ControlScalarType tMaterialPenalty = pow(tMaterialFraction, mPenaltyExponent);

            // out-of-plane thickness interpolation
            ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
            ControlScalarType tThicknessInterpolation = tThicknessTwo + 
              ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );

            // compute light-generated and dark current densities
            StateScalarType tCellElectricPotential = tInterpolateFromNodal(iCellOrdinal,tBasisValues,aState);
            ResultScalarType tLightCurrentDensity  = ( tMaterialPenalty * 
                tLightCurrentDensityModel(tCellElectricPotential) ) / tThicknessInterpolation;
            ResultScalarType tDarkCurrentDensity = 
                tDarkCurrentDensityModel(tCellElectricPotential) / tThicknessInterpolation;

            auto tWeight = tCubWeights(iGpOrdinal) * tDetJ;
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
            {
                ResultScalarType tCellResult = tBasisValues(tFieldOrdinal) * tWeight * tThicknessInterpolation
                  * ( tCellElectricPotential * ( tLightCurrentDensity + tDarkCurrentDensity ) );
                Kokkos::atomic_add( &aResult(iCellOrdinal,tFieldOrdinal), tCellResult );
            }
        });
    }

private:
    void 
    setOutofPlaneThickness(
        Plato::MaterialModel<EvaluationType> & aMaterialModel
    )
    {
        std::vector<std::string> tThickness = aMaterialModel.property("out-of-plane thickness");
        if ( tThickness.empty() )
        {
            auto tMsg = std::string("Array of out-of-plane thicknesses is empty. ") 
              + "Dark current density cannnot be computed.";
            ANALYZE_THROWERR(tMsg)
        }
        else
        {
            mOutofPlaneThickness.clear();
            for(size_t tIndex = 0; tIndex < tThickness.size(); tIndex++)
            {
              Plato::Scalar tMyThickness = std::stod(tThickness[tIndex]);
              mOutofPlaneThickness.push_back(tMyThickness);
            }
        }
    }
};

template<typename EvaluationType>
class VolumeTwoPhaseAlloy : public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr int mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;
 
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    Plato::Scalar mPenaltyExponent = 3.0;

    std::vector<Plato::Scalar> mOutofPlaneThickness;

public:
    VolumeTwoPhaseAlloy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aParamList, aFuncName)
    {
        this->initialize(aParamList);
    }
    ~VolumeTwoPhaseAlloy(){}

    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        // out-of-plane thicknesses for two-phase electrical material
        Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
        Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();

        // get basis functions and weights associated with the integration rule
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // evaluate volume for a two-phase electrical material
        auto tNumCells = mSpatialDomain.numCells();
        Kokkos::parallel_for("compute volume", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            // evaluate cell jacobian
            auto tCubPoint  = tCubPoints(iGpOrdinal);
            auto tCubWeight = tCubWeights(iGpOrdinal);
            auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

            // compute cell volume
            ResultScalarType tCellVolume = Plato::determinant(tJacobian);
            tCellVolume *= tCubWeight;

            // evaluate out-of-plane thickness interpolation function
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ControlScalarType tDensity = 
                Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
            ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
            ControlScalarType tThicknessInterpolation = tThicknessTwo + 
              ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
            
            // apply penalty to volume
            ResultScalarType tPenalizedVolume = tThicknessInterpolation * tCellVolume;

            Kokkos::atomic_add(&aResult(iCellOrdinal), tPenalizedVolume);
        });
    }

private:
    void 
    initialize(
        Teuchos::ParameterList & aParamList
    )
    {
        Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
        auto tMaterialModel = tMaterialFactory.create(mSpatialDomain.getMaterialName());
        this->setOutofPlaneThickness(tMaterialModel.operator*());

        mPenaltyExponent = aParamList.get<Plato::Scalar>("Penalty Exponent", 3.0);
    }

    void 
    setOutofPlaneThickness(
        Plato::MaterialModel<EvaluationType> & aMaterialModel
    )
    {
        std::vector<std::string> tThickness = aMaterialModel.property("out-of-plane thickness");
        if ( tThickness.empty() )
        {
            auto tMsg = std::string("Array of out-of-plane thicknesses is empty. ") 
              + "Volume criterion for two-phase material alloy cannnot be computed.";
            ANALYZE_THROWERR(tMsg)
        }
        else
        {
            mOutofPlaneThickness.clear();
            for(size_t tIndex = 0; tIndex < tThickness.size(); tIndex++)
            {
              Plato::Scalar tMyThickness = std::stod(tThickness[tIndex]);
              mOutofPlaneThickness.push_back(tMyThickness);
            }
        }
    }
};

/******************************************************************************/
template<typename EvaluationType>
class ElectrostaticsResidual : public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr int mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;

    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

    // to be derived from a new body load class in the future
    std::shared_ptr<Plato::SingleDiodeTwoPhaseAlloy<EvaluationType>> mSingleDiode; 
    std::shared_ptr<Plato::NaturalBCs<ElementType, mNumDofsPerNode>> mSurfaceLoads;

    std::vector<std::string> mPlottable;

public:
    ElectrostaticsResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList
    ) : 
      FunctionBaseType(aSpatialDomain,aDataMap)
    {
        // obligatory: define dof names in order
        mDofNames.push_back("electric_potential");

        // create material constitutive model
        Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
        mMaterialModel = tMaterialFactory.create(aSpatialDomain.getMaterialName());

        this->parseCurrentDensity();
        // TODO: create surface loads

        auto tResidualParams = aParamList.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }
    ~ElectrostaticsResidual(){}

    Plato::Solutions 
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    { return aSolutions; }

    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
        
        // inline functors
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
        Plato::GeneralFluxDivergence<ElementType> tComputeDivergence;
        Plato::ScalarGrad<ElementType>            tComputeScalarGrad;

        // interpolate nodal values to integration points
        Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;

        // integration rules
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();   

        // quantity of interests
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigScalarType>      
          tVolume("InterpolateFromNodalvolume",tNumCells);
        Plato::ScalarMultiVectorT<GradScalarType>   
          tElectricField("electrical field", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultScalarType> 
          tCurrentDensity("current density", tNumCells, mNumSpatialDims);
        Plato::ScalarArray4DT<ResultScalarType> 
          tMaterialTensor("material tensor", tNumCells, tNumPoints, mNumSpatialDims, mNumSpatialDims);
        
        // evaluate material tensor
        mMaterialModel->computeMaterialTensor(mSpatialDomain,aState,aControl,tMaterialTensor);

        // evaluate internal forces       
        Kokkos::parallel_for("evaluate electrostatics residual", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tCellVolume(0.0);
  
            Plato::Array<mNumSpatialDims,GradScalarType>   tCellElectricField(0.0);
            Plato::Array<mNumSpatialDims,ResultScalarType> tCellCurrentDensity(0.0);
            Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
  
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            // compute electrical field 
            tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tCellVolume);
            tComputeScalarGrad(iCellOrdinal,tCellElectricField,aState,tGradient);

            // compute current density
            for (Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
              tCellCurrentDensity(tDimI) = 0.0;
              for (Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
                tCellCurrentDensity(tDimI) += tMaterialTensor(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) 
                  * tCellElectricField(tDimJ);
              }
            }

            // apply divergence operator to current density
            tCellVolume *= tCubWeights(iGpOrdinal);
            tComputeDivergence(iCellOrdinal,aResult,tCellCurrentDensity,tGradient,tCellVolume,-1.0);
          
            for(Plato::OrdinalType tIndex=0; tIndex<mNumSpatialDims; tIndex++)
            {
                Kokkos::atomic_add(&tElectricField(iCellOrdinal,tIndex),  -1.0*tCellVolume*tCellElectricField(tIndex));
                Kokkos::atomic_add(&tCurrentDensity(iCellOrdinal,tIndex), tCellVolume*tCellCurrentDensity(tIndex));
            }
            Kokkos::atomic_add(&tVolume(iCellOrdinal),tCellVolume);
        });

        // evaluate volume forces
        if( mSingleDiode != nullptr )
        {
          mSingleDiode->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
        }

        Kokkos::parallel_for("compute cell quantities", 
          Kokkos::RangePolicy<>(0, tNumCells),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tIndex=0; tIndex<mNumSpatialDims; tIndex++)
            {
                tElectricField(iCellOrdinal,tIndex)  /= tVolume(iCellOrdinal);
                tCurrentDensity(iCellOrdinal,tIndex) /= tVolume(iCellOrdinal);
            }
        });

        if( std::count(mPlottable.begin(),mPlottable.end(),"electrical field") ) 
        { toMap(mDataMap, tElectricField, "electrical field", mSpatialDomain); }
        if( std::count(mPlottable.begin(),mPlottable.end(),"current density" ) )
        { toMap(mDataMap, tCurrentDensity, "current density" , mSpatialDomain); }
    }

    void
    evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        // add contributions from natural boundary conditions
        if( mSurfaceLoads != nullptr )
        {
            mSurfaceLoads->get(aSpatialModel,aState,aControl,aConfig,aResult,1.0);
        }
    }

private:
    void parseCurrentDensity(
        Teuchos::ParameterList & aParamList
    )
    {
        if(aParamList.isSublist("Current Density"))
        {
            // to be derived from a volume force class in the future
            auto tMaterialName = mSpatialDomain.getMaterialName();
            mSingleDiode = std::make_shared<Plato::SingleDiodeTwoPhaseAlloy<EvaluationType>>(
                    tMaterialName,aParamList.sublist("Single Diode")
                );
        }
    }
};

}

/******************************************************************************/
/*! Base class for electrical element
*/
/******************************************************************************/
template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class ElementElectrical : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumNodesPerFace;
    using TopoElementTypeT::mNumSpatialDims;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumDofsPerNode = 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
};
// class ElementElectrical 

namespace FactoryElectrical
{

/******************************************************************************//**
 * \brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
};

}
// FactoryElectrical

/******************************************************************************//**
 * \brief Concrete class for use as the physics template argument in Plato::Elliptic::Problem
**********************************************************************************/
template<typename TopoElementType>
class Electrical
{
public:
    typedef Plato::FactoryElectrical::FunctionFactory FunctionFactory;
    using ElementType = ElementElectrical<TopoElementType>;
};
// Electrical

}

namespace ElectrostaticsTest
{

   Teuchos::RCP<Teuchos::ParameterList> tGenericParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                            \n"
    "<ParameterList name='Spatial Model'>                                                                          \n"
      "<ParameterList name='Domains'>                                                                              \n"
        "<ParameterList name='Design Volume'>                                                                      \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                            \n"
          "<Parameter name='Material Model' type='string' value='Mystic'/>                                         \n"
        "</ParameterList>                                                                                          \n"
      "</ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Material Models'>                                                                        \n"
      "<ParameterList name='Mystic'>                                                                               \n"
        "<ParameterList name='Two Phase Electrical Conductivity'>                                                  \n"
          "<Parameter  name='Material Name'            type='Array(string)' value='{silver,aluminum}'/>            \n"
          "<Parameter  name='Electrical Conductivity'  type='Array(double)' value='{0.15,0.25}'/>                  \n"
          "<Parameter  name='Out-of-Plane Thickness'   type='Array(double)' value='{0.12,0.22}'/>                  \n"
        "</ParameterList>                                                                                          \n"
      "</ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Criteria'>                                                                               \n"
    "  <ParameterList name='Objective'>                                                                            \n"
    "    <Parameter name='Type' type='string' value='Weighted Sum'/>                                               \n"
    "    <Parameter name='Functions' type='Array(string)' value='{My Power}'/>                                     \n"
    "    <Parameter name='Weights' type='Array(double)' value='{1.0}'/>                                            \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Power'>                                                                             \n"
    "    <Parameter name='Type'                        type='string'        value='Scalar Function'/>              \n"
    "    <Parameter name='Scalar Function Type'        type='string'        value='Strength Constraint'/>          \n"
    "    <Parameter name='Exponent'                    type='double'        value='2.0'/>                          \n"
    "    <Parameter name='Minimum Value'               type='double'        value='1.0e-6'/>                       \n"
    "  </ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Source Terms'>                                                                           \n"
    "  <ParameterList name='Single Diode'>                                                                         \n"
    "    <Parameter name='Functions' type='Array(string)' value='{My Dark CD ,My Light-Generated CD}'/>            \n"
    "    <Parameter name='Weights'   type='Array(double)' value='{1.0,1.0}'/>                                      \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Dark CD'>                                                                           \n"
    "    <Parameter  name='Function'        type='string'      value='Two Phase Dark Current Density'/>            \n"
    "    <Parameter  name='Type'            type='string'      value='Nonlinear'/>                                 \n"
    "    <Parameter  name='Model'           type='string'      value='Quadratic'/>           ,                     \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Light-Generated CD'>                                                                \n"
    "    <Parameter  name='Function'        type='string'      value='Two Phase Light-Generated Current Density'/> \n"
    "    <Parameter  name='Type'            type='string'      value='Linear'/>                                    \n"
    "    <Parameter  name='Model'           type='string'      value='Constant'/>                                  \n"
      "</ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
  "</ParameterList>                                                                                                \n"
  );

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialElectricalConductivity_Error)
{
    // TEST ONE: ERROR
    Teuchos::RCP<Teuchos::ParameterList> tParamListError = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                                  \n"
        "<ParameterList name='Spatial Model'>                                                                \n"
          "<ParameterList name='Domains'>                                                                    \n"
            "<ParameterList name='Design Volume'>                                                            \n"
              "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
            "</ParameterList>                                                                                \n"
          "</ParameterList>                                                                                  \n"
        "</ParameterList>                                                                                    \n"
        "<ParameterList name='Material Models'>                                                              \n"
          "<ParameterList name='Mystic'>                                                                     \n"
            "<ParameterList name='Isotropic Linear Elastic'>                                                 \n"
              "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
              "<Parameter  name='Youngs Modulus' type='double' value='4.0'/>                                 \n"
              "<Parameter  name='Mass Density'   type='double' value='0.5'/>                                 \n"
            "</ParameterList>                                                                                \n"
          "</ParameterList>                                                                                  \n"
        "</ParameterList>                                                                                    \n"
       "</ParameterList>                                                                                     \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElementElectrical<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamListError.operator*());
    TEST_THROW(tFactoryMaterial.create("Mystic"),std::runtime_error);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialElectricalConductivity)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                                  \n"
        "<ParameterList name='Spatial Model'>                                                                \n"
          "<ParameterList name='Domains'>                                                                    \n"
            "<ParameterList name='Design Volume'>                                                            \n"
              "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
            "</ParameterList>                                                                                \n"
          "</ParameterList>                                                                                  \n"
        "</ParameterList>                                                                                    \n"
        "<ParameterList name='Material Models'>                                                              \n"
          "<ParameterList name='Mystic'>                                                                     \n"
            "<ParameterList name='Electrical Conductivity'>                                                  \n"
              "<Parameter  name='Electrical Conductivity' type='double' value='0.35'/>                       \n"
            "</ParameterList>                                                                                \n"
          "</ParameterList>                                                                                  \n"
        "</ParameterList>                                                                                    \n"
       "</ParameterList>                                                                                     \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElementElectrical<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamList.operator*());
    auto tMaterial = tFactoryMaterial.create("Mystic");
    auto tElectricalConductivity = tMaterial->property("electrical conductivity");
    auto tScalarValue = std::stod(tElectricalConductivity.back());
    TEST_FLOATING_EQUALITY(0.35,tScalarValue,1e-6);
    TEST_THROW(tMaterial->property("electrical_conductivity"),std::runtime_error);

    std::vector<std::vector<Plato::Scalar>> tGold = {{0.35,0.},{0.,0.35}};
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::TensorConstant<tNumSpaceDims> tTensor = tMaterial->getTensorConstant("material tensor");
    for(Plato::OrdinalType tDimI = 0; tDimI < tNumSpaceDims; tDimI++){
        for(decltype(tDimI) tDimJ = 0; tDimJ < tNumSpaceDims; tDimJ++){
            TEST_FLOATING_EQUALITY(tGold[tDimI][tDimJ],tTensor(tDimI,tDimJ),1e-6);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialDielectric)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                  \n"
        "<ParameterList name='Spatial Model'>                                                \n"
          "<ParameterList name='Domains'>                                                    \n"
            "<ParameterList name='Design Volume'>                                            \n"
              "<Parameter name='Element Block' type='string' value='body'/>                  \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>               \n"
            "</ParameterList>                                                                \n"
          "</ParameterList>                                                                  \n"
        "</ParameterList>                                                                    \n"
        "<ParameterList name='Material Models'>                                              \n"
          "<ParameterList name='Mystic'>                                                     \n"
            "<ParameterList name='Dielectric'>                                               \n"
              "<Parameter  name='Electrical Constant'          type='double' value='0.15'/>  \n"
              "<Parameter  name='Relative Static Permittivity' type='double' value='0.35'/>  \n"
            "</ParameterList>                                                                \n"
          "</ParameterList>                                                                  \n"
        "</ParameterList>                                                                    \n"
       "</ParameterList>                                                                     \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElementElectrical<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamList.operator*());
    auto tMaterial = tFactoryMaterial.create("Mystic");
    auto tElectricalConstant = tMaterial->property("electrical constant");
    auto tScalarValue = std::stod(tElectricalConstant.back());
    TEST_FLOATING_EQUALITY(0.15,tScalarValue,1e-6);
    auto tRelativeStaticPermittivity = tMaterial->property("Relative Static Permittivity");
    tScalarValue = std::stod(tRelativeStaticPermittivity.back());
    TEST_FLOATING_EQUALITY(0.35,tScalarValue,1e-6);

    std::vector<std::vector<Plato::Scalar>> tGold = {{0.0525,0.},{0.,0.0525}};
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::TensorConstant<tNumSpaceDims> tTensor = tMaterial->getTensorConstant("material tensor");
    for(Plato::OrdinalType tDimI = 0; tDimI < tNumSpaceDims; tDimI++){
        for(decltype(tDimI) tDimJ = 0; tDimJ < tNumSpaceDims; tDimJ++){
            TEST_FLOATING_EQUALITY(tGold[tDimI][tDimJ],tTensor(tDimI,tDimJ),1e-6);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialElectricalConductivityTwoPhaseAlloy)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                            \n"
        "<ParameterList name='Spatial Model'>                                                          \n"
          "<ParameterList name='Domains'>                                                              \n"
            "<ParameterList name='Design Volume'>                                                      \n"
              "<Parameter name='Element Block' type='string' value='body'/>                            \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>                         \n"
            "</ParameterList>                                                                          \n"
          "</ParameterList>                                                                            \n"
        "</ParameterList>                                                                              \n"
        "<ParameterList name='Material Models'>                                                        \n"
          "<ParameterList name='Mystic'>                                                               \n"
            "<ParameterList name='Two Phase Electrical Conductivity'>                                  \n"
              "<Parameter  name='Electrical Conductivity'  type='Array(double)' value='{0.15, 0.25}'/> \n"
              "<Parameter  name='Out-of-Plane Thickness'   type='Array(double)' value='{0.12, 0.22}'/> \n"
            "</ParameterList>                                                                          \n"
          "</ParameterList>                                                                            \n"
        "</ParameterList>                                                                              \n"
       "</ParameterList>                                                                               \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElementElectrical<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamList.operator*());
    auto tMaterial = tFactoryMaterial.create("Mystic");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DarkCurrentDensityQuadraticFit)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                       \n"
        "<ParameterList name='Spatial Model'>                                                     \n"
          "<ParameterList name='Domains'>                                                         \n"
            "<ParameterList name='Design Volume'>                                                 \n"
              "<Parameter name='Element Block' type='string' value='body'/>                       \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>                    \n"
            "</ParameterList>                                                                     \n"
          "</ParameterList>                                                                       \n"
        "</ParameterList>                                                                         \n"
        "<ParameterList name='Source Terms'>                                                      \n"
          "<ParameterList name='Dark Current Density'>                                            \n"
            "<Parameter  name='Model'              type='string'   value='Custom Quadratic Fit'/> \n"
            "<Parameter  name='Performance Limit'  type='double'   value='-0.22'/>                \n"
            "<Parameter  name='a'                  type='double'   value='0.0'/>                  \n"
            "<Parameter  name='b'                  type='double'   value='1.27E-06'/>             \n"
            "<Parameter  name='c'                  type='double'   value='25.94253'/>             \n"
            "<Parameter  name='m1'                 type='double'   value='0.38886'/>              \n"
            "<Parameter  name='b1'                 type='double'   value='0.0'/>                  \n"
            "<Parameter  name='m2'                 type='double'   value='30.0'/>                 \n"
            "<Parameter  name='b2'                 type='double'   value='6.520373'/>             \n"
          "</ParameterList>                                                                       \n"
        "</ParameterList>                                                                         \n"
       "</ParameterList>                                                                          \n"
      );

    // TEST ONE: V > 0
    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElementElectrical<Plato::Tri3>>::Residual;
    Plato::DarkCurrentDensityQuadraticFit<Residual,Plato::Scalar> 
      tCurrentDensityModel("Dark Current Density",tParamList.operator*());
    Residual::StateScalarType tElectricPotential = 0.67186;
    Plato::Scalar tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    Plato::Scalar tTol = 1e-4;
    TEST_FLOATING_EQUALITY(47.1463,tDarkCurrentDensity,tTol);

    // TEST 2: V = 0
    tElectricPotential = 0.;
    tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    TEST_FLOATING_EQUALITY(0.,tDarkCurrentDensity,tTol);
    
    // TEST 3: -0.22 < V < 0 
    tElectricPotential = -0.06189;
    tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    TEST_FLOATING_EQUALITY(-0.0240665,tDarkCurrentDensity,tTol);

    // TEST 4: V < -0.22 
    tElectricPotential = -0.25;
    tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    TEST_FLOATING_EQUALITY(-0.979627,tDarkCurrentDensity,tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LightGeneratedCurrentDensityConstant)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                       \n"
        "<ParameterList name='Spatial Model'>                                                     \n"
          "<ParameterList name='Domains'>                                                         \n"
            "<ParameterList name='Design Volume'>                                                 \n"
              "<Parameter name='Element Block' type='string' value='body'/>                       \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>                    \n"
            "</ParameterList>                                                                     \n"
          "</ParameterList>                                                                       \n"
        "</ParameterList>                                                                         \n"
        "<ParameterList name='Source Terms'>                                                      \n"
          "<ParameterList name='Light-Generated Current Density'>                                 \n"
            "<Parameter  name='Model'              type='string'   value='Constant'/>             \n"
            "<Parameter  name='Generation Rate'    type='double'   value='0.5'/>                  \n"
            "<Parameter  name='Illumination Power' type='double'   value='10.0'/>                 \n"
          "</ParameterList>                                                                       \n"
        "</ParameterList>                                                                         \n"
       "</ParameterList>                                                                          \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElementElectrical<Plato::Tri3>>::Residual;
    Plato::LightGeneratedCurrentDensityConstant<Residual,Plato::Scalar> 
      tCurrentDensityModel("Light-Generated Current Density",tParamList.operator*());
    Residual::StateScalarType tElectricPotential = 0.67186;
    Plato::Scalar tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    Plato::Scalar tTol = 1e-4;
    TEST_FLOATING_EQUALITY(5.,tDarkCurrentDensity,tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LightCurrentDensityTwoPhaseAlloy)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::ElementElectrical<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT   = typename Residual::StateScalarType;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    
    // create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);

    // create current density
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    TEST_ASSERT(tGenericParamList->isSublist("Source Terms") == true);
    Plato::LightCurrentDensityTwoPhaseAlloy<Residual> 
      tCurrentDensity("Mystic","My Light-Generated CD",tGenericParamList.operator*());
    
    // create result/output workset
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultWS("result workset", tNumCells, tDofsPerCell);
    tCurrentDensity.evaluate(tOnlyDomainDefined,tStateWS,tControlWS,tConfigWS,tResultWS,1.0);

    // test against gold
    auto tHost = Kokkos::create_mirror_view(tResultWS);
    Plato::Scalar tTol = 1e-6;
    std::vector<std::vector<Plato::Scalar>>tGold = {{-41.078313,-41.078313,-41.078313},
                                                    {-41.078313,-41.078313,-41.078313}};
    Kokkos::deep_copy(tHost, tResultWS);
    for(Plato::OrdinalType i = 0; i < tNumCells; i++){
      for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
        TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
      }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DarkCurrentDensityTwoPhaseAlloy)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::ElementElectrical<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT   = typename Residual::StateScalarType;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    
    // create state workset
    Plato::ScalarVector tState("States", tNumVerts);
    Plato::blas1::fill(0.67186, tState);
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);

    // create current density
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    TEST_ASSERT(tGenericParamList->isSublist("Source Terms") == true);
    Plato::DarkCurrentDensityTwoPhaseAlloy<Residual> 
      tCurrentDensity("Mystic","My Dark CD",tGenericParamList.operator*());
    
    // create result/output workset
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultWS("result workset", tNumCells, tDofsPerCell);
    tCurrentDensity.evaluate(tOnlyDomainDefined,tStateWS,tControlWS,tConfigWS,tResultWS,1.0);

    // test against gold
    auto tHost = Kokkos::create_mirror_view(tResultWS);
    Plato::Scalar tTol = 1e-6;
    std::vector<std::vector<Plato::Scalar>>tGold = {{37.8684771,37.8684771,37.8684771},
                                                    {37.8684771,37.8684771,37.8684771}};
    Kokkos::deep_copy(tHost, tResultWS);
    for(Plato::OrdinalType i = 0; i < tNumCells; i++){
      for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
        TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
      }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SingleDiode)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::ElementElectrical<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT   = typename Residual::StateScalarType;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    
    // create state workset
    Plato::ScalarVector tState("States", tNumVerts);
    Plato::blas1::fill(0.67186, tState);
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);

    // create current density
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    TEST_ASSERT(tGenericParamList->isSublist("Source Terms") == true);
    Plato::SingleDiodeTwoPhaseAlloy<Residual> tSingleDiode("Mystic",tGenericParamList.operator*());
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultWS("result workset", tNumCells, tDofsPerCell);
    tSingleDiode.evaluate(tOnlyDomainDefined,tStateWS,tControlWS,tConfigWS,tResultWS,1.0/*scale*/);

    // test against gold
    auto tHost = Kokkos::create_mirror_view(tResultWS);
    Plato::Scalar tTol = 1e-6;
    std::vector<std::vector<Plato::Scalar>>tGold = {{-3.2098359,-3.2098359,-3.2098359},
                                                    {-3.2098359,-3.2098359,-3.2098359}};
    Kokkos::deep_copy(tHost, tResultWS);
    for(Plato::OrdinalType i = 0; i < tNumCells; i++){
      for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
        TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
      }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PowerSurfaceDensity)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::ElementElectrical<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT   = typename Residual::StateScalarType;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    
    // create state workset
    Plato::ScalarVector tState("States", tNumVerts);
    Plato::blas1::fill(0.67186, tState);
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);

    // create current density
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
}

}