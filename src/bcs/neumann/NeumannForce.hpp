/*
 * NeumannForce.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "FadTypes.hpp"
#include "SpatialModel.hpp"
#include "SurfaceArea.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM and UNIFORM COMPONENT.
 *
 * \tparam ElementType  Element type (e.g., MechanicsElement<Tet10>)
 * \tparam mNumDofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<typename EvaluationType,
         Plato::OrdinalType NumForceDof=EvaluationType::ElementType::mNumDofsPerNode,
         Plato::OrdinalType DofOffset=0>
class NeumannForce
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  /// @brief side set name
  const std::string mSideSetName;
  /// @brief force values
  const Plato::Array<NumForceDof> mFlux;

public:
  NeumannForce(
    const std::string               & aSideSetName, 
    const Plato::Array<NumForceDof> & aFlux
  );

  void operator()(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aScale,
          Plato::Scalar         aCycle
  ) const;
};
// class NeumannForce

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
NeumannForce<EvaluationType,NumForceDof,DofOffset>::
NeumannForce(
  const std::string               & aSideSetName, 
  const Plato::Array<NumForceDof> & aFlux
) :
  mSideSetName(aSideSetName),
  mFlux(aFlux)
{
}
// class NeumannForce::NeumannForce

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void 
NeumannForce<EvaluationType,NumForceDof,DofOffset>::
operator()(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aScale,
        Plato::Scalar         aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));

  auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
  auto tNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
  Plato::OrdinalType tNumFaces = tElementOrds.size();

  Plato::SurfaceArea<ElementType> surfaceArea;

  auto tFlux = mFlux;
  auto tCubatureWeights = ElementType::Face::getCubWeights();
  auto tCubaturePoints  = ElementType::Face::getCubPoints();
  auto tNumPoints = tCubatureWeights.size();

  Kokkos::parallel_for("surface integral",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
  {
    auto tElementOrdinal = tElementOrds(aSideOrdinal);
    Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
    for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
    {
      tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
    }
    auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
    auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
    auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
    auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);
    ResultScalarType tSurfaceArea(0.0);
    surfaceArea(tElementOrdinal, tLocalNodeOrds, tBasisGrads, tConfigWS, tSurfaceArea);
    tSurfaceArea *= aScale;
    tSurfaceArea *= tCubatureWeight;
    // project into result workset
    for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
    {
      for( Plato::OrdinalType tDof=0; tDof<NumForceDof; tDof++)
      {
        auto tElementDofOrdinal = tLocalNodeOrds[tNode] * mNumDofsPerNode + tDof + DofOffset;
        ResultScalarType tResult = tBasisValues(tNode)*tFlux[tDof]*tSurfaceArea;
        Kokkos::atomic_add(&tResultWS(tElementOrdinal,tElementDofOrdinal), tResult);
      }
    }
  });
}
// class NeumannForce::operator()

}
// namespace Plato
