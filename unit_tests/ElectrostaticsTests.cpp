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
#include "Analyze_Diagnostics.hpp"
#include "util/PlatoTestHelpers.hpp"

// plato
#include "Tri3.hpp"
#include "Simp.hpp"
#include "ToMap.hpp"
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

#include "elliptic/electrical/ElectricalElement.hpp"
#include "elliptic/electrical/SupportedOptionEnums.hpp"
#include "elliptic/electrical/FactoryElectricalMaterial.hpp"
#include "elliptic/electrical/DarkCurrentDensityQuadratic.hpp"
#include "elliptic/electrical/LightGeneratedCurrentDensityConstant.hpp"
#include "elliptic/electrical/DarkCurrentDensityTwoPhaseAlloy.hpp"
#include "elliptic/electrical/LightCurrentDensityTwoPhaseAlloy.hpp"
#include "elliptic/electrical/FactoryCurrentDensityEvaluator.hpp"

#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/AbstractVectorFunction.hpp"

namespace Plato
{




template<typename EvaluationType>
class SourceEvaluator
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
      const Plato::Scalar                                & aScale
  ) 
  const = 0;
};
// class SourceEvaluator

// TODO: create factory to allocate volume forces. it will require refactoring the interfaces 
// to take workset metadata; e.g., similar to how it is done in the Ifem unit test. 
template<typename EvaluationType>
class SourceWeightedSum : public Plato::SourceEvaluator<EvaluationType>
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
    SourceWeightedSum(
      const std::string            & aMaterialName,
            Teuchos::ParameterList & aParamList
    ) : 
      mMaterialName(aMaterialName)
    {
      this->initialize(aParamList);
    }
    ~SourceWeightedSum(){}

    void 
    evaluate(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
        const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
        const Plato::Scalar                                & aScale
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
      if( !tSourceTermsParamList.isSublist("Source") ){
        auto tMsg = std::string("Parameter is not valid. Argument ('Source') is not a parameter list");
        ANALYZE_THROWERR(tMsg)
      }
      auto tSourceParamList = tSourceTermsParamList.sublist("Source");
      this->parseFunctions(tSourceParamList);
      this->parseWeights(tSourceParamList);

      this->createCurrentDensityEvaluators(aParamList);
    }

    void parseFunctions(
      Teuchos::ParameterList & aParamList
    )
    {
      bool tIsArray = aParamList.isType<Teuchos::Array<std::string>>("Functions");
      if( !tIsArray)
      {
        auto tMsg = std::string("Argument ('Functions') is not defined in ('Source') parameter list");
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
        Plato::FactoryCurrentDensityEvaluator<EvaluationType> tFactoryCurrentDensityEvaluator;
        std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> tEvaluator = 
          tFactoryCurrentDensityEvaluator.create(mMaterialName,tFunctionName,aParamList);
        mCurrentDensityEvaluators.push_back(tEvaluator);
      }
    }
};

namespace FactorySourceEvaluator
{

  template <typename EvaluationType>
  inline std::shared_ptr<Plato::SourceEvaluator<EvaluationType>> 
  create(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList
  )
  {
    if( !aParamList.isSublist("Source Terms") )
    {
      auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
      ANALYZE_THROWERR(tMsg)
    }
    auto tSourceTermsParamList = aParamList.sublist("Source Terms");
    if( !tSourceTermsParamList.isSublist("Source") )
    {
      auto tMsg = std::string("Parameter is not valid. Argument ('Source') is not a parameter list");
      ANALYZE_THROWERR(tMsg)
    }
    auto tSourceParamList = tSourceTermsParamList.sublist("Source");
    if( !tSourceParamList.isParameter("Type") )
    {
      auto tMsg = std::string("Parameter ('Type') is not defined in parameter list ('Source'), ") 
        + "source evaluator cannot be determined";
      ANALYZE_THROWERR(tMsg)
    }
    Plato::electrical::SourceEvaluatorEnum tS2E;
    auto tType = tSourceParamList.get<std::string>("Type");
    auto tLowerType = Plato::tolower(tType);
    auto tSupportedSourceEvaluatorEnum = tS2E.get(tLowerType);
    switch (tSupportedSourceEvaluatorEnum)
    {
      case Plato::electrical::source_evaluator::WEIGHTED_SUM:
        return std::make_shared<Plato::SourceWeightedSum<EvaluationType>>(aMaterialName,aParamList);
        break;
      default:
        return nullptr;
        break;
    }
  }

}
// namespace FactorySourceEvaluator

namespace Elliptic
{

template<typename EvaluationType>
class CriterionPowerSurfaceDensityTwoPhase : public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
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
    
    std::string mCriterionFunctionName = "";

    Plato::Scalar mPenaltyExponent = 3.0;
    Plato::Scalar mMinErsatzMaterialValue = 0.0;
    std::vector<Plato::Scalar> mOutofPlaneThickness;

    std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> mCurrentDensityEvaluator;

public:
    CriterionPowerSurfaceDensityTwoPhase(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
        const std::string            & aFuncName
    ) : 
      FunctionBaseType(aSpatialDomain, aDataMap, aParamList, aFuncName),
      mCriterionFunctionName(aFuncName)
    {
      this->initialize(aParamList);
    }
    ~CriterionPowerSurfaceDensityTwoPhase(){}

  void 
  evaluate(
      const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType> & aControl,
      const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>  & aConfig,
            Plato::ScalarVectorT      <typename EvaluationType::ResultScalarType>  & aResult,
            Plato::Scalar                                                            aCycle = 1.0
  ) override
  { 
    this->evaluate_conditional(aState, aControl, aConfig, aResult, aCycle);
  }

  void
  evaluate_conditional(
      const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
      const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
            Plato::ScalarVectorT      <ResultScalarType>  & aResult,
            Plato::Scalar                                   aCycle = 1.0
  ) const override
  {
    // integration rule
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints  = tCubWeights.size();
    // out-of-plane thicknesses
    Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
    Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();
    // build local functor
    Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;
    // evaluate current density
    Plato::Scalar tScale = aCycle;
    Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
    Plato::ScalarMultiVectorT<ResultScalarType> tCurrentDensity("current density",tNumCells,tNumPoints);
    mCurrentDensityEvaluator->evaluate(aState,aControl,aConfig,tCurrentDensity,tScale);
    // evaluate dark current density
    Kokkos::parallel_for("power surface density", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        // get basis functions and weights for this integration point
        auto tCubPoint = tCubPoints(iGpOrdinal);
        auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        // out-of-plane thickness interpolation
        ControlScalarType tDensity =
          Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
        ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
        ControlScalarType tThicknessInterpolation = tThicknessTwo + 
          ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
        // evaluate electric potential
        StateScalarType tCellElectricPotential = 
          tInterpolateFromNodal(iCellOrdinal,tBasisValues,aState);
        ResultScalarType tWeightedCurrentDensity = 
          tCurrentDensity(iCellOrdinal,iGpOrdinal) * tCubWeights(iGpOrdinal) * tDetJ;
        for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
        {
            ResultScalarType tCellResult = tBasisValues(tFieldOrdinal) * 
              tCellElectricPotential * tWeightedCurrentDensity * tThicknessInterpolation;
            Kokkos::atomic_add( &aResult(iCellOrdinal), tCellResult );
        }
    });
  }

private:
  void initialize(
    Teuchos::ParameterList & aParamList
  )
  {
    // build current density function
    this->buildCurrentDensityFunction(aParamList);
    // set out-of-plane thickness 
    Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
    auto tMaterialModel = tMaterialFactory.create(mSpatialDomain.getMaterialName());
    this->setOutofPlaneThickness(tMaterialModel.operator*());
  }

  void buildCurrentDensityFunction(
    Teuchos::ParameterList & aParamList
  )
  {
    if( !aParamList.isSublist("Criteria") ){
      auto tMsg = std::string("Parameter is not valid. Argument ('Criteria') is not a parameter list");
      ANALYZE_THROWERR(tMsg)
    }
    auto tCriteriaParamList = aParamList.sublist("Criteria");
    if( !tCriteriaParamList.isSublist(mCriterionFunctionName) ){
      auto tMsg = std::string("Parameter is not valid. Argument ('") 
        + mCriterionFunctionName + "') is not a parameter list";
      ANALYZE_THROWERR(tMsg)
    }
    auto tMyCriterionParamList = tCriteriaParamList.sublist(mCriterionFunctionName);
    if( !tMyCriterionParamList.isParameter("Function") ){
      auto tMsg = std::string("Parameter ('Function') is not defined in parameter list ('") 
        + mCriterionFunctionName + "'), power surface density cannot be evaluated";
      ANALYZE_THROWERR(tMsg)
    }
    std::string tMaterialName = mSpatialDomain.getMaterialName();
    std::string tCurrentDensityFunctionName = tMyCriterionParamList.get<std::string>("Function");
    Plato::FactoryCurrentDensityEvaluator<EvaluationType> tFactoryCurrentDensityEvaluator;
    mCurrentDensityEvaluator = 
      tFactoryCurrentDensityEvaluator.create(tMaterialName,tCurrentDensityFunctionName,aParamList
    );
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

template<typename EvaluationType>
class CriterionVolumeTwoPhase : public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
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
  CriterionVolumeTwoPhase(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aParamList,
      const std::string            & aFuncName
  ) :
      FunctionBaseType(aSpatialDomain, aDataMap, aParamList, aFuncName)
  {
      this->initialize(aParamList);
  }
  ~CriterionVolumeTwoPhase(){}

  void 
  evaluate(
    const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
    const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
          Plato::ScalarVectorT      <ResultScalarType>  & aResult,
          Plato::Scalar                                   aCycle = 1.0
  ) override
  {
    this->evaluate_conditional(aState,aControl,aConfig,aResult,aCycle);
  }
  void
  evaluate_conditional(
      const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
      const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
            Plato::ScalarVectorT      <ResultScalarType>  & aResult,
            Plato::Scalar                                   aCycle = 1.0
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
class SteadyStateCurrentResidual : public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
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
    std::shared_ptr<Plato::SourceEvaluator<EvaluationType>> mSourceEvaluator; 
    std::shared_ptr<Plato::NaturalBCs<ElementType, mNumDofsPerNode>> mSurfaceLoads;

    std::vector<std::string> mPlottable;

public:
    SteadyStateCurrentResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList
    ) : 
      FunctionBaseType(aSpatialDomain,aDataMap)
    {
        this->initialize(aParamList);
    }
    ~SteadyStateCurrentResidual(){}

    Plato::Solutions 
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    { return aSolutions; }

    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar                                   aTimeStep = 1.0
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
            tComputeDivergence(iCellOrdinal,aResult,tCellCurrentDensity,tGradient,tCellVolume,1.0);

            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
              for (Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
              }
            }
          
            for(Plato::OrdinalType tIndex=0; tIndex<mNumSpatialDims; tIndex++)
            {
              // compute the electric field E = -\nabla{\phi} (or -\phi_{,j}, where j=1,\dots,dims)
              Kokkos::atomic_add(&tElectricField(iCellOrdinal,tIndex),  -1.0*tCellVolume*tCellElectricField(tIndex));
              // Ohm constitutive law J = -\gamma_{ij}\phi_{,j}, where \phi is the scalar electric potential, 
              // \gamma is the second order electric conductivity tensor, and J is the current density
              Kokkos::atomic_add(&tCurrentDensity(iCellOrdinal,tIndex), -1.0*tCellVolume*tCellCurrentDensity(tIndex));
            }
            Kokkos::atomic_add(&tVolume(iCellOrdinal),tCellVolume);
        });

        // evaluate volume forces
        if( mSourceEvaluator != nullptr )
        {
          mSourceEvaluator->evaluate( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
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

        if( std::count(mPlottable.begin(),mPlottable.end(),"electric field") ) 
        { Plato::toMap(mDataMap, tElectricField, "electric field", mSpatialDomain); }
        if( std::count(mPlottable.begin(),mPlottable.end(),"current density" ) )
        { Plato::toMap(mDataMap, tCurrentDensity, "current density" , mSpatialDomain); }
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
  void initialize(
    Teuchos::ParameterList & aParamList
  )
  {
    // obligatory: define dof names in order
    mDofNames.push_back("electric_potential");

    // create material constitutive model
    auto tMaterialName = mSpatialDomain.getMaterialName();
    Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
    mMaterialModel = tMaterialFactory.create(tMaterialName);

    // create source evaluator
    mSourceEvaluator = Plato::FactorySourceEvaluator::create<EvaluationType>(tMaterialName,aParamList);

    // parse output QoI plot table
    auto tResidualParams = aParamList.sublist("Output");
    if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
    {
        mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }
  }
    
};

}

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
    using ElementType = ElectricalElement<TopoElementType>;
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
    "    <Parameter name='Functions' type='Array(string)' value='{My Dark Power,My Light}'/>                       \n"
    "    <Parameter name='Weights' type='Array(double)' value='{1.0,1.0}'/>                                        \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Dark Power'>                                                                        \n"
    "    <Parameter name='Type'                   type='string'   value='Scalar Function'/>                        \n"
    "    <Parameter name='Scalar Function Type'   type='string'   value='Power Surface Density'/>                  \n"
    "    <Parameter name='Function'               type='string'   value='My Dark CD'/>                             \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Light Power'>                                                                       \n"
    "    <Parameter name='Type'                   type='string'   value='Scalar Function'/>                        \n"
    "    <Parameter name='Scalar Function Type'   type='string'   value='Power Surface Density'/>                  \n"
    "    <Parameter name='Function'               type='string'   value='My Light-Generated CD'/>                  \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Volume'>                                                                            \n"
    "    <Parameter name='Type'                   type='string'   value='Scalar Function'/>                        \n"
    "    <Parameter name='Scalar Function Type'   type='string'   value='Two Phase Volume'/>                       \n"
    "  </ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Source Terms'>                                                                           \n"
    "  <ParameterList name='Source'>                                                                               \n"
    "    <Parameter name='Type'      type='string'        value='Weighted Sum'/>                                   \n"
    "    <Parameter name='Functions' type='Array(string)' value='{My Dark CD ,My Light-Generated CD}'/>            \n"
    "    <Parameter name='Weights'   type='Array(double)' value='{1.0,1.0}'/>                                      \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Dark CD'>                                                                           \n"
    "    <Parameter  name='Function'        type='string'      value='Two Phase Dark Current Density'/>            \n"
    "    <Parameter  name='Model'           type='string'      value='Quadratic'/>           ,                     \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Light-Generated CD'>                                                                \n"
    "    <Parameter  name='Function'        type='string'      value='Two Phase Light-Generated Current Density'/> \n"
    "    <Parameter  name='Model'           type='string'      value='Constant'/>                                  \n"
      "</ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Output'>                                                                                 \n"
    "  <Parameter name='Plottable' type='Array(string)' value='{electrical field,current density}'/>              \n"
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

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
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

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
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

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
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

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamList.operator*());
    auto tMaterial = tFactoryMaterial.create("Mystic");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DarkCurrentDensityQuadratic)
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
    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
    Plato::DarkCurrentDensityQuadratic<Residual,Plato::Scalar> 
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

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
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
    using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;

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
    using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;

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
    using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;

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
    Plato::SourceWeightedSum<Residual> tSingleDiode("Mystic",tGenericParamList.operator*());
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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CriterionPowerSurfaceDensityTwoPhase)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
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
  Plato::Elliptic::CriterionPowerSurfaceDensityTwoPhase<Residual>
    tCriterionPowerSurfaceDensityTwoPhase(tOnlyDomainDefined,tDataMap,*tGenericParamList,"My Dark Power");
  Plato::ScalarVectorT<ResultT> tResultWS("result workset", tNumCells);
  tCriterionPowerSurfaceDensityTwoPhase.evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);
  // test against gold
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  Plato::Scalar tTol = 1e-6;
  std::vector<Plato::Scalar>tGold = {15.8378409629,15.8378409629};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    TEST_FLOATING_EQUALITY(tGold[i],tHost(i),tTol);
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CriterionVolumeTwoPhase)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
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
  Plato::Elliptic::CriterionVolumeTwoPhase<Residual>
    tCriterionVolumeTwoPhase(tOnlyDomainDefined,tDataMap,*tGenericParamList,"My Volume");
  Plato::ScalarVectorT<ResultT> tResultWS("result workset", tNumCells);
  tCriterionVolumeTwoPhase.evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);
  // test against gold
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  Plato::Scalar tTol = 1e-6;
  std::vector<Plato::Scalar>tGold = {0.10375,0.10375};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    TEST_FLOATING_EQUALITY(tGold[i],tHost(i),tTol);
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SteadyStateCurrentResidual_ConstantPotential)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
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
  Plato::Elliptic::SteadyStateCurrentResidual<Residual> 
    tResidual(tOnlyDomainDefined,tDataMap,tGenericParamList.operator*());
  Plato::ScalarMultiVectorT<ResultT> tResultWS("result",tNumCells,tNodesPerCell);
  tResidual.evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);

  // test against gold - electric field is zero due to constant electric potential; 
  // thus, internal forces are zero and the residual is equal to minus external forces
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  Plato::Scalar tTol = 1e-6;
  std::vector<std::vector<Plato::Scalar>>tGold = {{3.2098359,3.2098359,3.2098359},
                                                  {3.2098359,3.2098359,3.2098359}};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
      TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
    }
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SteadyStateCurrentResidual_NonConstantPotential)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
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
  auto tHostState = Kokkos::create_mirror_view(tState);
  for(Plato::OrdinalType i = 0; i < tNumVerts; i++)
  { tHostState(i) = tHostState(i) + (i*1e-2); }
  Kokkos::deep_copy(tState, tHostState);
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  
  // create spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  // create current density
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  Plato::Elliptic::SteadyStateCurrentResidual<Residual> 
    tResidual(tOnlyDomainDefined,tDataMap,tGenericParamList.operator*());
  Plato::ScalarMultiVectorT<ResultT> tResultWS("result",tNumCells,tNodesPerCell);
  tResidual.evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);

  // test against gold - electric field is zero due to constant electric potential; 
  // thus, internal forces are zero and the residual is equal to minus external forces
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  Plato::Scalar tTol = 1e-6;
  std::vector<std::vector<Plato::Scalar>>tGold = {{-17.276113,-17.272551,-17.272551},
                                                  {-12.440948,-12.437385,-12.440948}};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
      TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
    }
  }
}

}