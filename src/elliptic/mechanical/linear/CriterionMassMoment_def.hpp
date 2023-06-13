#pragma once

#include "BLAS1.hpp"
#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "Assembly.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
CriterionMassMoment<EvaluationType>::
CriterionMassMoment(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap, 
        Teuchos::ParameterList & aInputParams,
        std::string              aFuncName
) :
  FunctionBaseType(aFuncName, aSpatialDomain, aDataMap, aInputParams)
{
  this->initialize(aSpatialDomain,aInputParams);
}

template<typename EvaluationType>
CriterionMassMoment<EvaluationType>::
CriterionMassMoment(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap& aDataMap
) :
  FunctionBaseType("CriterionMassMoment", aSpatialDomain, aDataMap)
{}

template<typename EvaluationType>
void 
CriterionMassMoment<EvaluationType>::
initialize(
  const Plato::SpatialDomain   & aSpatialDomain,
        Teuchos::ParameterList & aInputParams
)
{
  this->parseMaterialDensity(aSpatialDomain,aInputParams);
  this->parseNormalizeCriterion(aSpatialDomain,aInputParams);
  this->computeTotalStructuralMass();
}

template<typename EvaluationType>
void 
CriterionMassMoment<EvaluationType>::
parseMaterialDensity(
  const Plato::SpatialDomain   & aSpatialDomain,
        Teuchos::ParameterList & aInputParams
)
{
  auto tMaterialName = aSpatialDomain.getMaterialName();
  auto tMaterialModels = aInputParams.get<Teuchos::ParameterList>("Material Models");
  const Teuchos::ParameterList &tMaterialInputs = tMaterialModels.sublist(tMaterialName);
  if(tMaterialInputs.isSublist(tMaterialName))
  {
    auto tMsg = std::string("Parameter list for material with name '") 
      + tMaterialName + "' is not defined";
    ANALYZE_THROWERR(tMsg)
  }
  Teuchos::ParameterList::ConstIterator tMaterialItr = tMaterialInputs.begin();
  const std::string &tMaterialModelType = tMaterialInputs.name(tMaterialItr);
  const Teuchos::ParameterEntry &tMaterialEntry = tMaterialInputs.entry(tMaterialItr);
  if(!tMaterialEntry.isList())
  {
    auto tMsg = std::string("Parameter entry in Material Models block is invalid. Parameter ") 
      + tMaterialModelType + "' is not a parameter list.";
    ANALYZE_THROWERR(tMsg)
  }
  const Teuchos::ParameterList& tMaterialModelInputs = tMaterialInputs.sublist(tMaterialModelType);
  if( !tMaterialModelInputs.isParameter("Mass Density") )
  {
    auto tMsg = std::string("Parameter 'Mass Density' is not defined for material with name '")
      + tMaterialName + "' is not defined. Total structural mass cannot be computed.";
    ANALYZE_THROWERR(tMsg)
  }
  mMassDensity = tMaterialModelInputs.get<Plato::Scalar>("Mass Density");
  if(mMassDensity <= 0.)
  {
    auto tMsg = std::string("Unphysical 'Mass Density' parameter specified, 'Mass Density' is set to '") 
      + std::to_string(mMassDensity) + ".";
    ANALYZE_THROWERR(tMsg)
  }      
}

template<typename EvaluationType>
void 
CriterionMassMoment<EvaluationType>::
parseNormalizeCriterion(
  const Plato::SpatialDomain   & aSpatialDomain,
        Teuchos::ParameterList & aInputParams
)
{
  const std::string tFunctionName = this->getName();
  Teuchos::ParameterList &tProblemParams = aInputParams.sublist("Criteria").sublist(tFunctionName);
  mNormalizeCriterion = tProblemParams.get<bool>("Normalize Criterion",false);
}

template<typename EvaluationType>
void
CriterionMassMoment<EvaluationType>::
computeTotalStructuralMass()
{
  if( !mNormalizeCriterion )
  { return; }
  auto tNumCells = mSpatialDomain.numCells();
  Plato::NodeCoordinate<mNumSpatialDims, mNumNodesPerCell> tCoordinates(mSpatialDomain.Mesh);
  Plato::ScalarArray3D tConfig("configuration", tNumCells, mNumNodesPerCell, mNumSpatialDims);
  Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>(tNumCells, tCoordinates, tConfig);
  Plato::ScalarVector tTotalStructuralMass("total mass", tNumCells);
  Plato::ScalarMultiVector tDensities("densities", tNumCells, mNumNodesPerCell);
  Kokkos::deep_copy(tDensities, 1.0);
  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();
  auto tMassDensity = mMassDensity;
  Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    auto tJacobian = ElementType::jacobian(tCubPoint, tConfig, iCellOrdinal);
    auto tVolume = Plato::determinant(tJacobian);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, tDensities);
    auto tLocalCellMass = tCellMass * tMassDensity * tVolume * tCubWeight;
    Kokkos::atomic_add(&tTotalStructuralMass(iCellOrdinal), tLocalCellMass);
  });
  Plato::blas1::local_sum(tTotalStructuralMass, mTotalStructuralMass);
}

template<typename EvaluationType>
void 
CriterionMassMoment<EvaluationType>::
setMaterialDensity(
  const Plato::Scalar aMaterialDensity
)
{
  mMassDensity = aMaterialDensity;
}

template<typename EvaluationType>
void 
CriterionMassMoment<EvaluationType>::
setCalculationType(
  const std::string & aCalculationType
)
{
  mCalculationType = Plato::tolower(aCalculationType);
}

template<typename EvaluationType>
bool 
CriterionMassMoment<EvaluationType>::
isLinear() 
const
{
  return true;
}

template<typename EvaluationType>
void
CriterionMassMoment<EvaluationType>::
evaluateConditional(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
  Plato::ScalarVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarVectorT<ResultScalarType>>(aWorkSets.get("result"));
  // evaluate criterion
  if (mCalculationType == "mass")
    computeStructuralMass(tControlWS, tConfigWS, tResultWS, aCycle);
  else if (mCalculationType == "firstx")
    computeFirstMoment(tControlWS, tConfigWS, tResultWS, 0, aCycle);
  else if (mCalculationType == "firsty")
    computeFirstMoment(tControlWS, tConfigWS, tResultWS, 1, aCycle);
  else if (mCalculationType == "firstz")
    computeFirstMoment(tControlWS, tConfigWS, tResultWS, 2, aCycle);
  else if (mCalculationType == "secondxx")
    computeSecondMoment(tControlWS, tConfigWS, tResultWS, 0, 0, aCycle);
  else if (mCalculationType == "secondyy")
    computeSecondMoment(tControlWS, tConfigWS, tResultWS, 1, 1, aCycle);
  else if (mCalculationType == "secondzz")
    computeSecondMoment(tControlWS, tConfigWS, tResultWS, 2, 2, aCycle);
  else if (mCalculationType == "secondxy")
    computeSecondMoment(tControlWS, tConfigWS, tResultWS, 0, 1, aCycle);
  else if (mCalculationType == "secondxz")
    computeSecondMoment(tControlWS, tConfigWS, tResultWS, 0, 2, aCycle);
  else if (mCalculationType == "secondyz")
    computeSecondMoment(tControlWS, tConfigWS, tResultWS, 1, 2, aCycle);
  else {
    ANALYZE_THROWERR("Specified mass moment calculation type not implemented.")
  }
}

template<typename EvaluationType>
void
CriterionMassMoment<EvaluationType>::
computeStructuralMass(
  const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
  const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
        Plato::ScalarVectorT      <ResultScalarType>  & aResult,
        Plato::Scalar aCycle
) const
{
  auto tNumCells = mSpatialDomain.numCells();
  auto tMassDensity = mMassDensity;
  auto tTotalStructuralMass = mTotalStructuralMass;
  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();
  Kokkos::parallel_for("structural mass", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);
    ResultScalarType tVolume = Plato::determinant(tJacobian);
    tVolume *= tCubWeight;
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControl);
    ResultScalarType tNormalizedCellMass = (tCellMass / tTotalStructuralMass) * tMassDensity * tVolume;
    Kokkos::atomic_add(&aResult(iCellOrdinal), tNormalizedCellMass);
  });
}

template<typename EvaluationType>
void
CriterionMassMoment<EvaluationType>::
computeFirstMoment(
  const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
  const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
        Plato::ScalarVectorT      <ResultScalarType>  & aResult,
        Plato::OrdinalType  aComponent,
        Plato::Scalar       aCycle
) const 
{
  assert(aComponent < mNumSpatialDims);
  auto tNumCells = mSpatialDomain.numCells();
  auto tMassDensity = mMassDensity;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  Plato::ScalarArray3DT<ConfigScalarType> tMappedPoints("mapped quad points", tNumCells, tNumPoints, mNumSpatialDims);
  mapQuadraturePoints(aConfig, tMappedPoints);
  Kokkos::parallel_for("first moment calculation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);
    ResultScalarType tVolume = Plato::determinant(tJacobian);
    tVolume *= tCubWeight;
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControl);
    ConfigScalarType tMomentArm = tMappedPoints(iCellOrdinal, iGpOrdinal, aComponent);
    Kokkos::atomic_add(&aResult(iCellOrdinal), tCellMass * tMassDensity * tVolume *tMomentArm);
  });
}

template<typename EvaluationType>
void
CriterionMassMoment<EvaluationType>::
computeSecondMoment(
  const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
  const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
        Plato::ScalarVectorT      <ResultScalarType>  & aResult,
        Plato::OrdinalType aComponent1,
        Plato::OrdinalType aComponent2,
        Plato::Scalar      aCycle
) const 
{
  assert(aComponent1 < mNumSpatialDims);
  assert(aComponent2 < mNumSpatialDims);
  auto tNumCells = mSpatialDomain.numCells();
  auto tMassDensity = mMassDensity;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  Plato::ScalarArray3DT<ConfigScalarType> tMappedPoints("mapped quad points", tNumCells, tNumPoints, mNumSpatialDims);
  mapQuadraturePoints(aConfig, tMappedPoints);
  Kokkos::parallel_for("second moment calculation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);
    ResultScalarType tVolume = Plato::determinant(tJacobian);
    tVolume *= tCubWeight;
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControl);
    ConfigScalarType tMomentArm1 = tMappedPoints(iCellOrdinal, iGpOrdinal, aComponent1);
    ConfigScalarType tMomentArm2 = tMappedPoints(iCellOrdinal, iGpOrdinal, aComponent2);
    ConfigScalarType tSecondMoment  = tMomentArm1 * tMomentArm2;
    Kokkos::atomic_add(&aResult(iCellOrdinal), tCellMass * tMassDensity * tVolume * tSecondMoment);
  });
}

template<typename EvaluationType>
void
CriterionMassMoment<EvaluationType>::
mapQuadraturePoints(
  const Plato::ScalarArray3DT <ConfigScalarType> & aConfig,
        Plato::ScalarArray3DT <ConfigScalarType> & aMappedPoints
) const
{
  auto tNumCells = mSpatialDomain.numCells();
  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();
  Kokkos::deep_copy(aMappedPoints, static_cast<ConfigScalarType>(0.0));
  Kokkos::parallel_for("map points", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint    = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    for (Plato::OrdinalType iDim=0; iDim<mNumSpatialDims; iDim++)
    {
      for (Plato::OrdinalType iNodeOrdinal=0; iNodeOrdinal<mNumNodesPerCell; iNodeOrdinal++)
      {
          aMappedPoints(iCellOrdinal, iGpOrdinal, iDim) += 
            tBasisValues(iNodeOrdinal) * aConfig(iCellOrdinal, iNodeOrdinal, iDim);
      }
    }
  });
}

} // namespace Elliptic

} // namespace Plato
