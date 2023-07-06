/*
 *  FactoryNeumannBC.hpp
 *
 *  Created on: July 5, 2023
 */

#pragma once

#include "base/SupportedParamOptions.hpp"
#include "bcs/neumann/SupportedParamOptions.hpp"

#include "bcs/neumann/NeumannForce.hpp"
#include "bcs/neumann/NeumannPressure.hpp"
#include "bcs/neumann/NeumannFollowerPressure.hpp"

namespace Plato
{

template<
  typename EvaluationType,
  Plato::OrdinalType NumForceDof=EvaluationType::ElementType::mNumDofsPerNode,
  Plato::OrdinalType DofOffset=0>
struct FactoryNeumannBC
{
public:
  std::shared_ptr<Plato::NeumannBoundaryConditionBase<NumForceDof>>
  create(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aSubList
  )
  {
    if( !aSubList.isParameter("Type") ){
      ANALYZE_THROWERR(std::string("ERROR: Input argument ('Type') is not defined in Neumann boundary condition ") + 
        "parameter list, " + "Neumann boundary condition cannot be constructed")
    }
    Plato::NeumannEnum tS2E;
    auto tType = aSubList.get<std::string>("Type");
    auto tSupportedEnum = tS2E.bc(tType);
    switch(tSupportedEnum) 
    {
      case Plato::neumann_bc::UNIFORM:
      case Plato::neumann_bc::UNIFORM_COMPONENT:
        return ( std::make_shared<Plato::NeumannForce<EvaluationType,NumForceDof,DofOffset>>(aParamList,aSubList) );
        break;
      case Plato::neumann_bc::UNIFORM_PRESSURE:  
        return ( std::make_shared<Plato::NeumannPressure<EvaluationType,NumForceDof,DofOffset>>(aParamList,aSubList) );
        break; 
      case Plato::neumann_bc::FOLLOWER_PRESSURE:
        return ( std::make_shared<Plato::NeumannFollowerPressure<EvaluationType,NumForceDof,DofOffset>>(
            aParamList,aSubList) 
        );
        break;
      default:
        return nullptr;
    }
  }
};

} // namespace Plato
