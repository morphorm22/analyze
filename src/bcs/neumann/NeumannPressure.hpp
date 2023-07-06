/*
 * NeumannPressure.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "WeightedNormalVector.hpp"
#include "bcs/neumann/NeumannBoundaryConditionBase.hpp"

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
class NeumannPressure : public Plato::NeumannBoundaryConditionBase<NumForceDof>
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
  // set local base class type
  using BaseClassType = Plato::NeumannBoundaryConditionBase<NumForceDof>;
  // set natural boundary condition base class member data
  using BaseClassType::mFlux;
  using BaseClassType::mSideSetName;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  /// @param [in] aSubList   neumann boundary condition parameter list
  NeumannPressure(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aSubList
  );

  /// @fn evaluate
  /// @brief evaluate neumann boundary condition - pressure
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
  ) const;

  /// @fn flux
  /// @brief update flux vector values
  /// @param [in] aFlux flux vector
  void
  flux(
    const Plato::Array<NumForceDof> & aFlux
  );
};
// class NeumannPressure

// function definitions

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
NeumannPressure<EvaluationType,NumForceDof,DofOffset>::
NeumannPressure(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aSubList
)
{
  if(!aSubList.isParameter("Sides")){
    ANALYZE_THROWERR(std::string("ERROR: Input argument ('Sides') is not defined in Neumann boundary condition ") +   
      "parameter list, side sets for Neumann boundary conditions cannot be determined")
  }
  mSideSetName = aSubList.get<std::string>("Sides");
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void 
NeumannPressure<EvaluationType,NumForceDof,DofOffset>::
evaluate(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle,
        Plato::Scalar         aScale
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
  Plato::WeightedNormalVector<ElementType> tComputeWeighthedNormalVector;

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
    Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
    for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
    {
      tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
    }
    auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
    auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
    auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
    auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);
    // area weighthed normal vector: (n/||n||)*||n||, i.e. normal vector is already multiplied by the surface area
    //
    Plato::Array<mNumSpatialDims, ConfigScalarType> tWeighthedNormalVector;
    tComputeWeighthedNormalVector(tElementOrdinal, tLocalNodeOrds, tBasisGrads, tConfigWS, tWeighthedNormalVector);
    // project into result workset
    for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
    {
      for( Plato::OrdinalType tDof=0; tDof<NumForceDof; tDof++)
      {
        auto tElementDofOrdinal = (tLocalNodeOrds[tNode] * mNumDofsPerNode) + tDof + DofOffset;
        ResultScalarType tVal = tWeighthedNormalVector(tDof) * tFlux(tDof) * aScale 
          * tCubatureWeight * tNormalMultiplier * tBasisValues(tNode);
        Kokkos::atomic_add(&tResultWS(tElementOrdinal, tElementDofOrdinal), tVal);
      }
    }
  });
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void 
NeumannPressure<EvaluationType,NumForceDof,DofOffset>::
flux(
  const Plato::Array<NumForceDof> & aFlux
)
{
  mFlux = aFlux;
}

}
// namespace Plato
