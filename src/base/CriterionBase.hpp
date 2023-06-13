/*
 *  CriterionBase.hpp
 *
 *  Created on: June 6, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "UtilsTeuchos.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/// @class CriterionBase
/// @brief criterion evaluator base class
class CriterionBase
{
protected:
  /// @brief contains mesh and model information
  const Plato::SpatialDomain & mSpatialDomain;
  /// @brief output database 
  Plato::DataMap & mDataMap;
  /// @brief criterion name
  const std::string mFunctionName;
  /// @brief include criterion in evaluation if true
  bool mCompute;       

public:
  /// @brief public typename for base criterion class
  using AbstractType = typename Plato::CriterionBase;

  /// @brief class constructor
  /// @param [in] aName      criterion name
  /// @param [in] aDomain    contains mesh and model information
  /// @param [in] aDataMap   output database
  /// @param [in] aParamList input problem parameters 
  CriterionBase(
    const std::string            & aName,
    const Plato::SpatialDomain   & aDomain,
         Plato::DataMap          & aDataMap,
         Teuchos::ParameterList  & aParamList
  ) :
    mSpatialDomain (aDomain),
    mDataMap       (aDataMap),
    mFunctionName  (aName),
    mCompute       (true)
  {
    this->initialize(aParamList);
  }

  /// @brief class constructor
  /// @param [in] aName    criterion name
  /// @param [in] aDomain  contains mesh and model information
  /// @param [in] aDataMap output database
  CriterionBase(
    const std::string            & aName,
    const Plato::SpatialDomain   & aDomain,
         Plato::DataMap          & aDataMap
  ) :
    mSpatialDomain (aDomain),
    mDataMap       (aDataMap),
    mFunctionName  (aName),
    mCompute       (true)
  {
  }

  /// @brief class destructor
  virtual ~CriterionBase(){}
  
  /// @fn domain
  /// @brief get spatial domain database - contains model information
  /// @return const reference to spatial domain
  decltype(mSpatialDomain) 
  domain() 
  const
  { return mSpatialDomain; }

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  virtual 
  bool 
  isLinear() 
  const = 0;

  /// @fn evaluateConditional
  /// @brief evaluate criterion
  /// @param [in,out] aWorkSets function domain and range workset database
  /// @param [in]     aCycle    scalar 
  virtual 
  void
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const = 0;  

  /// @fn evaluate
  /// @brief main evaluate criterion
  /// @param [in,out] aWorkSets function domain and range workset database
  /// @param [in]     aCycle    scalar 
  virtual 
  void
  evaluate(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  )
  { 
    if(mCompute){
      this->evaluateConditional(aWorkSets,aCycle); 
    }
  }

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aWorkSets function domain and range workset database
  /// @param [in] aCycle    scalar 
  virtual 
  void 
  updateProblem(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) 
  { return; }

  /// @fn getName
  /// @brief get criterion name
  /// @return string
  const decltype(mFunctionName)& 
  getName()
  { return mFunctionName; }

  /// @fn postEvaluate
  /// @brief post evaluate criterion
  /// @param [in,out] aGrad  criterion gradient
  /// @param [in,out] aValue criterion value
  virtual 
  void 
  postEvaluate(
    Plato::ScalarVector aGrad, 
    Plato::Scalar       aValue
  )
  { return; }

  /// @fn postEvaluate
  /// @brief post evaluate criterion
  /// @param [in] aValue criterion value 
  virtual 
  void 
  postEvaluate(
    Plato::Scalar& aValue
  )
  { return; }

  /// @fn setSpatialWeightFunction
  /// @brief set spatial weight function
  /// @param [in] aExpression mathematical expression
  virtual 
  void 
  setSpatialWeightFunction(
    std::string aExpression
  )
  { return; }

private:
  /// @fn initialize
  /// @brief initialize member data 
  /// @param aParamList input problem parameters
  void
  initialize(
    Teuchos::ParameterList & aParamList
  )
  {
    std::string tCurrentDomainName = mSpatialDomain.getDomainName();  
    auto tMyCriteria = aParamList.sublist("Criteria").sublist(mFunctionName);
    std::vector<std::string> tDomains = Plato::teuchos::parse_array<std::string>("Domains", tMyCriteria);
    if(tDomains.size() != 0)
    {
      mCompute = (std::find(tDomains.begin(), tDomains.end(), tCurrentDomainName) != tDomains.end());
      if(!mCompute)
      {
        std::stringstream ss;
        ss << "Block '" << tCurrentDomainName << "' will not be included in the calculation of '" 
          << mFunctionName << "'.";
        REPORT(ss.str());
      }
    }
  }
};
    
} // namespace Plato


