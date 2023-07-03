/*
 * NeumannPressure.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "SpatialModel.hpp"
#include "WeightedNormalVector.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM PRESSURE.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam mNumPhysicsDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam mNumDofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<typename EvaluationType,
         Plato::OrdinalType NumForceDof=EvaluationType::ElementType::mNumDofsPerNode,
         Plato::OrdinalType DofOffset=0>
class NeumannPressure
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
  /// @brief force magnitudes
  const Plato::Array<NumForceDof> mFlux;

public:
  NeumannPressure(
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
// class NeumannPressure

/***************************************************************************//**
 * \brief NeumannPressure::NeumannPressure constructor definition
*******************************************************************************/
template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
NeumannPressure<EvaluationType,NumForceDof,DofOffset>::
NeumannPressure(
  const std::string               & aSideSetName, 
  const Plato::Array<NumForceDof> & aFlux
) :
  mSideSetName(aSideSetName),
  mFlux(aFlux)
{}

/***************************************************************************//**
 * \brief NeumannPressure::operator() function definition
*******************************************************************************/
template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void 
NeumannPressure<EvaluationType,NumForceDof,DofOffset>::
operator()(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aScale,
        Plato::Scalar         aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));

  auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
  auto tNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
  auto tFaceOrds    = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);

  Plato::OrdinalType tNumFaces = tElementOrds.size();
  Plato::WeightedNormalVector<ElementType> weightedNormalVector;

  auto tFlux = mFlux;
  auto tCubatureWeights = ElementType::Face::getCubWeights();
  auto tCubaturePoints  = ElementType::Face::getCubPoints();
  auto tNumPoints = tCubatureWeights.size();

  // pressure forces should act towards the surface; thus, -1.0 is used to invert the outward facing normal inwards.
  Plato::Scalar tNormalMultiplier(-1.0);
  Kokkos::parallel_for("surface integral",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
  {
    auto tElementOrdinal = tElementOrds(aSideOrdinal);
    auto tElemFaceOrdinal = tFaceOrds(aSideOrdinal);
    Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
    for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
    {
      tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
    }
    auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
    auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
    auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
    auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);
    // compute area weighted normal vector
    Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVec;
    weightedNormalVector(tElementOrdinal, tLocalNodeOrds, tBasisGrads, tConfigWS, tWeightedNormalVec);
    // project into result workset
    for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
    {
      for( Plato::OrdinalType tDof=0; tDof<NumForceDof; tDof++)
      {
        auto tElementDofOrdinal = (tLocalNodeOrds[tNode] * mNumDofsPerNode) + tDof + DofOffset;
        ResultScalarType tVal = tWeightedNormalVec(tDof) * tFlux(tDof) * aScale 
          * tCubatureWeight * tNormalMultiplier * tBasisValues(tNode);
        Kokkos::atomic_add(&tResultWS(tElementOrdinal, tElementDofOrdinal), tVal);
      }
    }
  });
}
// class NeumannPressure::operator()

}
// namespace Plato
