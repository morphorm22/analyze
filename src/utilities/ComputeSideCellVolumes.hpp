/*
 * ComputeSideCellVolumes.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "MetaData.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/// @class ComputeSideCellVolumes
/// @brief compute volume of cells/elements along the surface
///
/// \f[
///   \Omega_{V}=\int_{\Omega}\det(J) d\Omega
/// \f]
///
/// where \f$J\f$ is the element Jacobian. 
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ComputeSideCellVolumes
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  /// @brief user assigned name for side set
  std::string mSideSetName;

public:
  /// @brief class constructor
  /// @param [in] aEntitySetName side set name
  ComputeSideCellVolumes(
    const std::string& aEntitySetName
  ) :
    mSideSetName(aEntitySetName)
  {}

  /// @fn operator()
  /// @brief compute cell volumes
  /// @param [in]     aSpatialModel    contains mesh and model information
  /// @param [in]     aWorkSets        range and domain database
  /// @param [in,out] aSideCellVolumes side cell volumes
  void 
  operator()(
    const Plato::SpatialModel                    & aSpatialModel,
    const Plato::WorkSets                        & aWorkSets,
          Plato::ScalarVectorT<ConfigScalarType> & aSideCellVolumes
  )
  {
    // get side set connectivity information
    auto tSideCellOrdinals = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
    // get input workset
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get body quadrature points and weights
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints  = tCubWeights.size();
    // compute volume of each cell in the entity set
    Kokkos::parallel_for("compute volume", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumSideCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType aGpOrdinal)
    {
      auto tCubPoint  = tCubPoints(aGpOrdinal);
      auto tCubWeight = tCubWeights(aGpOrdinal);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, tCellOrdinal);
      ConfigScalarType tVolume = Plato::determinant(tJacobian);
      tVolume *= tCubWeight;
      Kokkos::atomic_add(&aSideCellVolumes(aSideOrdinal), tVolume);
    });
  }
};

} // namespace Plato
