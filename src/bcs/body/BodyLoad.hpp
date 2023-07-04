/*
 *  BodyLoad.hpp
 *
 *  Created on: July 1, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "MetaData.hpp"
#include "SpatialModel.hpp"
#include "PlatoMeshExpr.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato
{

/// @brief body force evaluator
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class BodyLoad
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions 
  static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
  /// @brief number of degrees of freedom per node
  static constexpr Plato::OrdinalType mNumDofsPerNode = ElementType::mNumDofsPerNode;
  /// @brief number of nodes per cell
  static constexpr Plato::OrdinalType mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;

protected:
  /// @brief body force parameter sublit name
  const std::string mName;
  /// @brief spatial direction component 
  const Plato::OrdinalType mDof;
  /// @brief mathematical expression for the body force
  const std::string mFuncString;

public:
  /// @brief class constructor
  /// @param [in] aName  body load name
  /// @param [in] aParam body load parameter list
  BodyLoad(
    const std::string &aName, 
          Teuchos::ParameterList &aParam
  ) :
    mName(aName),
    mDof(aParam.get<Plato::OrdinalType>("Index", 0)),
    mFuncString(aParam.get<std::string>("Function"))
  {
  }
  /// @brief class destructor
  ~BodyLoad()
  {}

  /// @brief evaluate body load and update results workset (range workset)
  /// @param [in]     aSpatialDomain contains mesh and model information
  /// @param [in,out] aWorkSets      range and domain database
  /// @param [in]     aCycle         scalar cycle
  /// @param [in]     aScale         scalar multiplier
  void
  evaluate(
    const Plato::SpatialDomain & aSpatialDomain,
          Plato::WorkSets      & aWorkSets,
          Plato::Scalar          aCycle,
          Plato::Scalar          aScale
  ) const
  {
    // unpack worksets
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
    //
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints  = tCubWeights.size();
    // map points to physical space
    //
    Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
    Plato::ScalarArray3DT<ConfigScalarType> 
      tPhysicalPoints("cub points physical space", tNumCells, tNumPoints, mSpaceDim);
    Plato::mapPoints<ElementType>(tConfigWS, tPhysicalPoints);
    // get integrand values at quadrature points
    //
    Plato::ScalarMultiVectorT<ConfigScalarType> tFxnValues("function values", tNumCells*tNumPoints, 1);
    Plato::getFunctionValues<mSpaceDim>(tPhysicalPoints, mFuncString, tFxnValues);
    // integrate and assemble
    //
    auto tDof = mDof;
    Kokkos::parallel_for("compute body load", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      auto tCubPoint = tCubPoints(iGpOrdinal);
      auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal));
      auto tBasisValues = ElementType::basisValues(tCubPoint);
      ResultScalarType tDensity(0.0);
      for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < ElementType::mNumNodesPerCell; tFieldOrdinal++)
      {
        tDensity += tBasisValues(tFieldOrdinal)*tControlWS(iCellOrdinal, tFieldOrdinal);
      }
      auto tEntryOffset = iCellOrdinal * tNumPoints;
      auto tFxnValue = tFxnValues(tEntryOffset + iGpOrdinal, 0);
      auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
      for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < ElementType::mNumNodesPerCell; tFieldOrdinal++)
      {
        Kokkos::atomic_add(&tResultWS(iCellOrdinal,tFieldOrdinal*mNumDofsPerNode+tDof),
          tWeight * tFxnValue * tBasisValues(tFieldOrdinal) * tDensity);
      }
    });
  }

};
// end class BodyLoad

} // namespace Plato
