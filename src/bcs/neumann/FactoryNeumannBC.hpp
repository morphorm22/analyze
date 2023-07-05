/*
 *  FactoryNeumannBC.hpp
 *
 *  Created on: July 5, 2023
 */

#pragma once

#include "bcs/neumann/NeumannForce.hpp"
#include "bcs/neumann/NeumannPressure.hpp"

namespace Plato
{

enum struct neumann_bc
{
  UNIFORM = 1,
  UNIFORM_PRESSURE = 2,
  UNIFORM_COMPONENT = 3,
};

struct NeumannBCEnum
{
public:
  std::unordered_map<std::string,Plato::neumann_bc> s2e = 
  {
    {"uniform"          ,Plato::neumann_bc::UNIFORM},
    {"uniform pressure" ,Plato::neumann_bc::UNIFORM_PRESSURE},
    {"uniform component",Plato::neumann_bc::UNIFORM_COMPONENT},
  };

public:
  Plato::neumann_bc 
  bc(
    const std::string & aNBC
  ) const
  {
    auto tLowerNBC = Plato::tolower(aNBC);
    auto tItr = s2e.find(tLowerNBC);
    if( tItr == s2e.end() ){
      auto tMsg = this->getErrorMsg(tLowerNBC);
      ANALYZE_THROWERR(tMsg)
    }
    return tItr->second;
  }
private:
  std::string
  getErrorMsg(
    const std::string & aNBC
  ) const
  {
    auto tMsg = std::string("Did not find input Neumann boundary condition '") + aNBC
      + "'. Supported Neumann boundary conditions are: ";
    for(const auto& tPair : s2e){
      tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
  }
};

template<
  typename EvaluationType,
  Plato::OrdinalType NumForceDof=EvaluationType::ElementType::mNumDofsPerNode,
  Plato::OrdinalType DofOffset=0
>
struct FactoryNeumannBC
{
public:
  std::shared_ptr<Plato::NeumannBoundaryConditionBase<NumForceDof>>
  create(
    Teuchos::ParameterList & aSubList
  )
  {
    if( !aSubList.isParameter("Type") ){
      ANALYZE_THROWERR(std::string("ERROR: Input argument ('Type') is not defined in Neumann boundary condition ") + 
        "parameter list, " + "Neumann boundary condition cannot be constructed")
    }
    Plato::NeumannBCEnum tS2E;
    auto tType = aSubList.get<std::string>("Type");
    auto tSupportedEnum = tS2E.bc(tType);
    switch(tSupportedEnum)
    {
      case Plato::neumann_bc::UNIFORM:
      case Plato::neumann_bc::UNIFORM_COMPONENT:
      {
        return ( std::make_shared<Plato::NeumannForce<EvaluationType,NumForceDof,DofOffset>>(aSubList) );
        break;
      }
      case Plato::neumann_bc::UNIFORM_PRESSURE:
      {
        return ( std::make_shared<Plato::NeumannPressure<EvaluationType,NumForceDof,DofOffset>>(aSubList) );
        break;
      }
      default:
      {
        return nullptr;
      }
    }
}

};

} // namespace Plato
