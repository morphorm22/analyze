/*
 * ElastostaticsTotalLagrangianTests.cpp
 *
 *  Created on: May 10, 2023
 */

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// plato
#include "Tri3.hpp"
#include "WorksetBase.hpp"
#include "GradientMatrix.hpp"
#include "MechanicsElement.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

template<typename EvaluationType>
class StateGradient
{
private:
  using ElementType = typename EvaluationType::ElementType;

  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;

  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

public:
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::OrdinalType                                               & aCellIndex,
    const Plato::ScalarMultiVectorT<StateScalarType>                       & aStates,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad
  ) const
  {
    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
    {
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
      {
        Plato::OrdinalType tDof = (tNode * mNumDofsPerNode) + tDimI;
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
        {
          aStateGrad(tDimI,tDimJ) += aStates(aCellIndex,tDof) * aGradient(tNode,tDimJ);
        }
      }
    }
  }
};

template<typename EvaluationType>
class DeformationGradient
{
private:
  using ElementType = typename EvaluationType::ElementType;
  
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;

  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGrad
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        aDefGrad(tDimI,tDimJ) += aStateGrad(tDimI,tDimJ);
      }
      aDefGrad(tDimI,tDimI) += 1.0;
    }
  }
};

template<typename EvaluationType>
class RightDeformationTensor
{
private:
  using ElementType = typename EvaluationType::ElementType;
  
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;

  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

public:
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefTensor
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        aDefTensor(tDimI,tDimJ) += aDefGrad(tDimJ,tDimI) * aDefGrad(tDimI,tDimJ);
      }
    }
  }
};

template<typename EvaluationType>
class GreenLagrangeStrainTensor
{
private:
  using ElementType = typename EvaluationType::ElementType;
  
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;

  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

public:
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefTensor,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStrainTensor
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        aStrainTensor(tDimI,tDimJ) += 0.5 * aDefTensor(tDimI,tDimJ);
      }
      aStrainTensor(tDimI,tDimI) -= 0.5;
    }
  }
};

}

namespace ElastostaticsTotalLagrangianTests
{

TEUCHOS_UNIT_TEST( ElastostaticsTotalLagrangianTests, TBD )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Plato::RightDeformationTensor<Residual> tComputeRightDeformationTensor;
  Plato::GreenLagrangeStrainTensor<Residual> tGreenLagrangeStrainTensor;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGrad(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGrad);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGrad(StrainT(0.));
      tComputeDeformationGradient(tStateGrad,tDefGrad);
      // compute cauchy-green deformation tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefTensor(StrainT(0.));
      tComputeRightDeformationTensor(tDefGrad,tDefTensor);
      // compute green-lagrange strain tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStrainTensor(StrainT(0.));
      tGreenLagrangeStrainTensor(tDefTensor,tStrainTensor);
    });
}

}