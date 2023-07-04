#ifndef BODYLOADS_HPP
#define BODYLOADS_HPP

#include <Teuchos_ParameterList.hpp>

#include "PlatoTypes.hpp"
#include "SpatialModel.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "PlatoMeshExpr.hpp"

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
    Plato::ScalarArray3DT<ConfigScalarType> tPhysicalPoints("cub points physical space", tNumCells, tNumPoints, mSpaceDim);
    Plato::mapPoints<ElementType>(tConfigWS, tPhysicalPoints);
    // get integrand values at quadrature points
    //
    Plato::ScalarMultiVectorT<ConfigScalarType> tFxnValues("function values", tNumCells*tNumPoints, 1);
    Plato::getFunctionValues<mSpaceDim>(tPhysicalPoints, mFuncString, tFxnValues);
    // integrate and assemble
    //
    auto tDof = mDof;
    Plato::VectorEntryOrdinal<mSpaceDim, mSpaceDim> tVectorEntryOrdinal(aSpatialDomain.Mesh);
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

template<typename EvaluationType>
class BodyLoads
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @ list of body forces
  std::vector<std::shared_ptr<BodyLoad<EvaluationType>>> mBodyLoads;

public:
  BodyLoads(Teuchos::ParameterList &aParams) :
          mBodyLoads()
  {
    for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
    {
      const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
      const std::string &tName = aParams.name(tIndex);
      if(!tEntry.isList())
      {
        ANALYZE_THROWERR("ERROR: Parameter in Body Loads block not valid.  Expect lists only.");
      }
      Teuchos::ParameterList& tSublist = aParams.sublist(tName);
      std::shared_ptr<Plato::BodyLoad<EvaluationType>> tBodyLoad =
        std::make_shared<Plato::BodyLoad<EvaluationType>>(tName, tSublist);
      mBodyLoads.push_back(tBodyLoad);
    }
  }

  void
  get(
    const Plato::SpatialDomain & aSpatialDomain,
        Plato::WorkSets        & aWorkSets,
        Plato::Scalar            aCycle,
        Plato::Scalar            aScale = 1.0
  ) const
  {
    for(const auto & tBodyLoad : mBodyLoads){
      tBodyLoad->evaluate(aSpatialDomain, aWorkSets, aCycle, aScale);
    }
  }
};

}

#endif
