/*
 * ComputeSideCellVolumes.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "MetaData.hpp"
#include "SpatialModel.hpp"

#include "utilities/SurfaceArea.hpp"

namespace Plato
{

/// @class ComputeSideCellFaceAreas
/// @brief compute area of cells on side set
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ComputeSideCellFaceAreas
{
private:
  /// @brief topological element type for body and surface elements
  using BodyElementType = typename EvaluationType::ElementType;
  using FaceElementType = typename BodyElementType::Face;
  /// @brief number of nodes per face
  static constexpr auto mNumNodesPerFace = BodyElementType::mNumNodesPerFace;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  /// @brief user assigned side set name
  std::string mSideSetName;

public:
  /// @brief class constructor
  /// @param [in] aEntitySetName side set name
  ComputeSideCellFaceAreas(
    const std::string& aEntitySetName
  ) :
    mSideSetName(aEntitySetName)
  {}

  /// @fn operator()
  /// @brief compute cell volumes
  /// @param [in]     aSpatialModel      contains mesh and model information
  /// @param [in]     aWorkSets          range and domain database
  /// @param [in,out] aSideCellFaceAreas side cell areas
  void 
  operator()(
    const Plato::SpatialModel                    & aSpatialModel,
    const Plato::WorkSets                        & aWorkSets,
          Plato::ScalarVectorT<ConfigScalarType> & aSideCellFaceAreas)
  {
    // get side set connectivity information
    auto tSideCellOrdinals = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tLocalNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
    // get input workset
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // create surface area functor
    Plato::SurfaceArea<BodyElementType> tComputeFaceArea;
    // get face quadrature points and weights
    auto tFaceCubPoints    = FaceElementType::getCubPoints();
    auto tFaceCubWeights   = FaceElementType::getCubWeights();
    auto tFaceNumCubPoints = tFaceCubWeights.size();
    // compute characteristic length of each cell in the entity set
    Kokkos::parallel_for("compute volume", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumSideCells, tFaceNumCubPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType aGpOrdinal)
    {
      // get quadrature point and weight
      auto tFaceCubPoint   = tFaceCubPoints(aGpOrdinal);
      auto tFaceCubWeight  = tFaceCubWeights(aGpOrdinal);
      auto tFaceBasisGrads = FaceElementType::basisGrads(tFaceCubPoint);
      // get local node ordinal on boundary entity
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrdinals;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrdinals(tIndex) = tLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute entity area
      ConfigScalarType tFaceArea(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      tComputeFaceArea(tCellOrdinal,tFaceLocalNodeOrdinals,tFaceBasisGrads,tConfigWS,tFaceArea);
      tFaceArea *= tFaceCubWeight;
      // add characteristic length contribution from this quadrature point, i.e. Gauss point
      Kokkos::atomic_add(&aSideCellFaceAreas(aSideOrdinal), tFaceArea);
    });
  }
};

} // namespace Plato
