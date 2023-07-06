/*
 *  FollowerPressure.hpp
 *
 *  Created on: July 5, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "bcs/neumann/NeumannBoundaryConditionBase.hpp"
#include "elliptic/mechanical/nonlinear/FollowerPressure.hpp"
#include "elliptic/thermomechanics/nonlinear/FollowerPressure.hpp"

namespace Plato
{

template<
  typename EvaluationType,
  Plato::OrdinalType NumForceDof=EvaluationType::ElementType::mNumDofsPerNode,
  Plato::OrdinalType DofOffset=0>
class NeumannFollowerPressure : public Plato::NeumannBoundaryConditionBase<NumForceDof>
{
private:
  std::shared_ptr<Plato::NeumannBoundaryConditionBase<NumForceDof>> mFollowerPressure;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  /// @param [in] aSubList   neumann boundary condition parameter list
  NeumannFollowerPressure(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aSubList
  )
  {
    this->initialize(aParamList,aSubList);
  }

  /// @fn flux
  /// @brief update flux vector values
  /// @param [in] aFlux flux vector
  void 
  flux(
    const Plato::Array<NumForceDof> & aFlux
  )
  { 
    mFollowerPressure->flux(aFlux); 
  }

  /// @fn evaluate
  /// @brief evaluate follower pressure
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     range and domain database
  /// @param [in]     aCycle        scalar
  /// @param [in]     aScale        scalar
  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  ) const
  {
    mFollowerPressure->evaluate(aSpatialModel,aWorkSets,aCycle,aScale);
  }

private:
  void
  initialize(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aSubList
  )
  {
    Plato::PhysicsEnum tS2E;
    auto tPhysics = aParamList.get<std::string>("Physics");
    auto tEnumPhysics = tS2E.physics(tPhysics);
    switch (tEnumPhysics)
    {
      case Plato::physics_t::MECHANICAL:
        mFollowerPressure = 
          std::make_shared<Plato::mechanical::FollowerPressure<EvaluationType,NumForceDof,DofOffset>>(
            aParamList,aSubList
        ); 
        break;
      case Plato::physics_t::THERMOMECHANICAL:
        mFollowerPressure = 
          std::make_shared<Plato::thermomechanical::FollowerPressure<EvaluationType,NumForceDof,DofOffset>>(
          aParamList,aSubList
        );
        break;
      default:
        ANALYZE_THROWERR(std::string("ERROR: Neumann boundary conditon ('Follower Pressure') is not supported ") 
          + "for ('" + tPhysics + "') physics, supported options are: 'mechanical' and 'thermomechanical'")
        break;
    }
  }
};

} // namespace Plato