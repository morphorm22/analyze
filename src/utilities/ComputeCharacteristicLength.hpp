/*
 * ComputeCharacteristicLength.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "MetaData.hpp"
#include "SpatialModel.hpp"

#include "utilities/ComputeSideCellVolumes.hpp"
#include "utilities/ComputeSideCellFaceAreas.hpp"

namespace Plato
{

/// @class ComputeCharacteristicLength
/// @brief compute cell characteristic length \f$L=V/A\f$, where \f$V\f$ is the element volume, 
/// \f$A\f$ is the element area
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ComputeCharacteristicLength
{
private:
  /// @brief scalar types associated with the automatic differentation evaluation type
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  /// @brief user assigned side set name
  std::string mSideSetName;

public:
  /// @brief class constructor
  /// @param [in] aEntitySetName side set name 
  ComputeCharacteristicLength(
    const std::string& aEntitySetName
  ) :
    mSideSetName(aEntitySetName)
  {}
  
  /// @fn operator()
  /// @brief compute characteristic length of all the side cells 
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in]     aWorkSets     range and domain database
  /// @param [in,out] aCharLength   characteristic length
  void operator()(
    const Plato::SpatialModel                    & aSpatialModel,
    const Plato::WorkSets                        & aWorkSets,
          Plato::ScalarVectorT<ConfigScalarType> & aCharLength
  )
  {
    // get side set connectivity information
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tLocalNodeOrds     = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
    // compute volumes of cells in side
    Plato::ScalarVectorT<ConfigScalarType> tSideCellVolumes("volume",tNumSideCells);
    Plato::ComputeSideCellVolumes<EvaluationType> tComputeSideCellVolumes(mSideSetName);
    tComputeSideCellVolumes(aSpatialModel,aWorkSets,tSideCellVolumes);
    // compute face areas of cells in side
    Plato::ScalarVectorT<ConfigScalarType> tSideCellFaceAreas("area",tNumSideCells);
    Plato::ComputeSideCellFaceAreas<EvaluationType> tComputeSideCellFaceAreas(mSideSetName);
    tComputeSideCellFaceAreas(aSpatialModel,aWorkSets,tSideCellFaceAreas);
    // compute characteristic length of each cell in the entity set
    Kokkos::parallel_for("compute characteristic length", Kokkos::RangePolicy<>(0, tNumSideCells),
    KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal)
    {
      aCharLength(aSideOrdinal) = tSideCellVolumes(aSideOrdinal) / tSideCellFaceAreas(aSideOrdinal);
    });
  }
};

} // namespace Plato
