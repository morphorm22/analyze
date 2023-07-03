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
#include "bcs/neumann/NeumannForce.hpp"
#include "bcs/neumann/NeumannPressure.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Natural boundary condition type ENUM
*******************************************************************************/
struct Neumann
{
  enum bc_t
  {
    UNDEFINED = 0,
    UNIFORM = 1,
    UNIFORM_PRESSURE = 2,
    UNIFORM_COMPONENT = 3,
  };
};
// struct Neumann

/// @fn natural_boundary_condition_type
/// @brief return natural boundary conditions 
/// @param [in] aType input name string
/// @return natural boundary condition type
inline 
Plato::Neumann::bc_t 
natural_boundary_condition_type(
  const std::string & aType
)
{
  auto tLowerTag = Plato::tolower(aType);
  if(tLowerTag == "uniform")
  {
    return Plato::Neumann::UNIFORM;
  }
  else if(tLowerTag == "uniform pressure")
  {
    return Plato::Neumann::UNIFORM_PRESSURE;
  }
  else if(tLowerTag == "uniform component")
  {
    return Plato::Neumann::UNIFORM_COMPONENT;
  }
  else
  {
    ANALYZE_THROWERR(std::string("Natural Boundary Condition: 'Type' Parameter Keyword: '") 
      + tLowerTag + "' is not supported.")
  }
}
// function natural_boundary_condition_type

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
  /// @brief natural boundary condition type
  const std::string mType;
  /// @brief side set name
  const std::string mSideSetName;
  /// @brief force values
  Plato::Array<NumForceDof> mFlux;
  /// @brief expression evaluators
  std::shared_ptr<Plato::MathExpr> mFluxExpr[NumForceDof];

public:
  /// @brief class constructor
  /// @param [in] aLoadName side set name
  /// @param [in] aSubList  input parameter sublist name
  NeumannBC(
    const std::string            & aLoadName, 
          Teuchos::ParameterList & aSubList
  ) :
    mName(aLoadName),
    mType(aSubList.get<std::string>("Type")),
    mSideSetName(aSubList.get<std::string>("Sides")),
    mFluxExpr{nullptr}
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
  decltype(mSideSetName) const& 
  getSideSetName() 
  const 
  { return mSideSetName; }

}; // class NeumannBC

/***************************************************************************//**
 * \brief NeumannBC::get function definition
*******************************************************************************/
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
  for(int iDim=0; iDim<NumForceDof; iDim++)
  {
    if(mFluxExpr[iDim])
    {
      mFlux(iDim) = mFluxExpr[iDim]->value(aCycle);
    }
  }

  auto tType = Plato::natural_boundary_condition_type(mType);
  switch(tType)
  {
    case Plato::Neumann::UNIFORM:
    case Plato::Neumann::UNIFORM_COMPONENT:
    {
      Plato::NeumannForce<EvaluationType,NumForceDof,DofOffset> tSurfaceLoad(mSideSetName, mFlux);
      tSurfaceLoad(aSpatialModel, aWorkSets, aCycle, aScale);
      break;
    }
    case Plato::Neumann::UNIFORM_PRESSURE:
    {
      Plato::NeumannPressure<EvaluationType,NumForceDof,DofOffset> tSurfacePress(mSideSetName, mFlux);
      tSurfacePress(aSpatialModel, aWorkSets, aCycle, aScale);
      break;
    }
    default:
    {
      std::stringstream tMsg;
      tMsg << "Natural Boundary Condition: Natural Boundary Condition Type '" 
        << mType.c_str() << "' is NOT supported.";
      ANALYZE_THROWERR(tMsg.str().c_str())
    }
  }
}

}
// namespace Plato
