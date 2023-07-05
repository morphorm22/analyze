/*
 * NeumannBC.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include <sstream>

#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoMathExpr.hpp"
#include "PlatoUtilities.hpp"
#include "bcs/neumann/FactoryNeumannBC.hpp"
#include "bcs/neumann/NeumannBoundaryConditionBase.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for natural boundary conditions.
 *
 * \tparam ElementType  Element type
 * \tparam mNumDofsPerNode  number degrees of freedom per node
 * \tparam mNumDofsOffset    degrees of freedom offset
 *
*******************************************************************************/
template<typename EvaluationType,
         Plato::OrdinalType NumForceDof=EvaluationType::ElementType::mNumDofsPerNode,
         Plato::OrdinalType DofOffset=0>
class NeumannBC
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;
  /// @brief natural boundary condition name
  const std::string mName;
  /// @brief force values
  Plato::Array<NumForceDof> mFlux;
  /// @brief expression evaluators
  std::shared_ptr<Plato::MathExpr> mFluxExpr[NumForceDof];
  /// @brief neumann boundary condition evaluator
  std::shared_ptr<Plato::NeumannBoundaryConditionBase<NumForceDof>> mNeumannBC;

public:
  /// @brief class constructor
  /// @param [in] aLoadName side set name
  /// @param [in] aSubList  input parameter sublist name
  NeumannBC(
    const std::string            & aLoadName, 
          Teuchos::ParameterList & aSubList
  ) :
    mName(aLoadName),
    mFluxExpr{nullptr}
  {
    Plato::FactoryNeumannBC<EvaluationType,NumForceDof,DofOffset> tFactory;
    mNeumannBC = tFactory.create(aSubList);
    if(mNeumannBC == nullptr){
      ANALYZE_THROWERR(std::string("ERROR: Neumann boundary condition factory return a null pointer, ") 
        + "unsupported Neumann boundary condition requested!")
    }
    this->initialize(aSubList);
  }

  /// @brief class destructor
  ~NeumannBC(){}

  /// @brief evaluate natural boundary conditions
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     range and domain database
  /// @param [in]     aScale        scalar multiplier
  /// @param [in]     aCycle        scalar cycle
  void 
  get(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aScale,
          Plato::Scalar         aCycle
  );

  /// @fn getSubListName
  /// @brief get natural boundary condition parameter list name
  /// @return side set name string
  decltype(mName) const& 
  getSubListName() 
  const 
  { return mName; }

  /// @fn getSideSetName
  /// @brief get side set name
  /// @return side set name string
  std::string 
  getSideSetName() 
  const 
  { return (mNeumannBC->sideset()); }

private:
  void 
  initialize(
    Teuchos::ParameterList & aSubList
  );

}; // class NeumannBC

// function definitions

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void 
NeumannBC<EvaluationType,NumForceDof,DofOffset>::
initialize(
  Teuchos::ParameterList & aSubList
)
{
  auto tIsValue = aSubList.isType<Teuchos::Array<Plato::Scalar>>("Vector");
  auto tIsExpr  = aSubList.isType<Teuchos::Array<std::string>>("Vector");
  if (tIsValue)
  {
    auto tFlux = aSubList.get<Teuchos::Array<Plato::Scalar>>("Vector");
    for(Plato::OrdinalType tDof=0; tDof<NumForceDof; tDof++)
    {
      mFlux(tDof) = tFlux[tDof];
    }
  }
  else
  if (tIsExpr)
  {
    auto tExpr = aSubList.get<Teuchos::Array<std::string>>("Vector");
    for(Plato::OrdinalType tDof=0; tDof<NumForceDof; tDof++)
    {
      mFluxExpr[tDof] = std::make_shared<Plato::MathExpr>(tExpr[tDof]);
      mFlux(tDof) = mFluxExpr[tDof]->value(0.0);
    }
  }
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void 
NeumannBC<EvaluationType,NumForceDof,DofOffset>::
get(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aScale,
        Plato::Scalar         aCycle
)
{
  for(int iDim=0; iDim<NumForceDof; iDim++){
    if(mFluxExpr[iDim]){
      mFlux(iDim) = mFluxExpr[iDim]->value(aCycle);
    }
  }
  mNeumannBC->flux(mFlux);
  mNeumannBC->evaluate(aSpatialModel,aWorkSets,aScale,aCycle);
}

}
// namespace Plato
