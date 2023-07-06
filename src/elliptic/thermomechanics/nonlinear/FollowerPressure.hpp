/*
 *  FollowerPressure.hpp
 *
 *  Created on: July 5, 2023
 */

#pragma once

#include "bcs/neumann/NeumannBoundaryConditionBase.hpp"

#include "AnalyzeMacros.hpp"
#include "GradientMatrix.hpp"
#include "WeightedNormalVector.hpp"
#include "InterpolateFromNodal.hpp"

#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermalDeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermoElasticDeformationGradient.hpp"

namespace Plato
{

namespace thermomechanical
{

/// @class FollowerPressure
///
/// @brief evaluates mechanical follower pressure:
/// \[
///  p_{f}=-\int_{\Gamma^{0}_{\bar{p}}}\delta{u} J F_{ij}^{-T}(n_j \bar{p})\ d\Gamma^0,\quad{i,j}=1,\dots,N_{dim}
/// \]
/// where \f$p_f\f$ denotes the follower pressure, \f$\Gamma^0\f$ is the undeformed configuration,
/// \f$\Gamma^{0}_{\bar{p}}\f$ is the loaded surface on the undeformed configuration, \f$J=\det(F)\f$, 
/// \f$F_{ij}\f$ is the deformation gradient, and \f$\bar{p}\f$ denotes the pressure. The thermomechanical 
/// deformation gradient is computed as:
/// \[
///   F_{ij}=F_{ik}^{\theta}F_{kj}^{u}\quad{i,j,k}=1,\dots,N_{dim},
/// \]
/// where \f$F_{ij}^{\theta}\f$ is the thermal deformation gradient and \f$F_{ij}^{u}\f$ is the mechanical
/// deformation gradient.
///
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types 
/// @tparam NumForceDof    number of force degrees of freedom
/// @tparam DofOffset      number of degrees of freedom offset
template<
  typename EvaluationType,
  Plato::OrdinalType NumForceDof=EvaluationType::ElementType::mNumDofsPerNode,
  Plato::OrdinalType DofOffset=0>
class FollowerPressure : public Plato::NeumannBoundaryConditionBase<NumForceDof>
{
private:
  /// @brief topological element typenames for parent body and face elements
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent element face
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of temperature degrees of freedom per node
  static constexpr auto mNumNodeStatePerNode = BodyElementBase::mNumNodeStatePerNode;
  /// @brief number of integration points per face
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType     = typename EvaluationType::StateScalarType;
  using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
  using ResultScalarType    = typename EvaluationType::ResultScalarType;
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  using StrainScalarType    = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief set local typename for base class
  using BaseClassType = Plato::NeumannBoundaryConditionBase<NumForceDof>;
  /// @brief flux magnitude
  using BaseClassType::mFlux;
  /// @brief side set name 
  using BaseClassType::mSideSetName;
  /// @brief input problem parameters
  Teuchos::ParameterList & mParamList;
  /// @brief material name
  std::string mMaterialName;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  /// @param [in] aSubList   neumann boundary condition parameter list
  FollowerPressure(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aSubList
  ) : 
    mParamList(aParamList)
  {
    this->initialize(aSubList);
  }

  /// @fn flux
  /// @brief update flux vector values
  /// @param [in] aFlux flux vector
  void 
  flux(
    const Plato::Array<NumForceDof> & aFlux
  )
  { mFlux = aFlux; }

  /// @brief evaluate thermo-mechanical follower pressure
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
  ) const
  {
    // unpack worksets
    //
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<NodeStateScalarType> tTempWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<NodeStateScalarType>>(aWorkSets.get("node states"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    // local functors
    //
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    Plato::InterpolateFromNodal<BodyElementBase,mNumNodeStatePerNode> tInterpolateFromNodal;
    Plato::ThermoElasticDeformationGradient<EvaluationType> tComputeThermoElasticDeformationGradient;
    Plato::ThermalDeformationGradient<EvaluationType> tComputeThermalDeformationGradient(mMaterialName,mParamList);
    // get side set information
    //
    auto tSideSetLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideSetLocalElemOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideSetLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // get parent body and face element integration rules
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // pressure acts towards the surface; therefore, -1.0 is used to invert the outward facing normal inwards
    //
    auto tFlux = mFlux;
    Plato::Scalar tNormalMultiplier(-1.0);
    Plato::OrdinalType tNumCellsOnSideSet = tSideSetLocalElemOrds.size();
    Kokkos::parallel_for("follower mechanical pressure",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      // quadrature point on parent body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideSetLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnParentBodyElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnParentBodyElemSurface(tIndex);
      }
      // get quadrature weights and basis functions on parent body element surface of interest
      //
      auto tCubWeightOnParentBodyElemSurface   = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisGradsOnParentBodyElemSurface  = BodyElementBase::basisGrads(tCubPointOnParentBodyElemSurface);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      // get node ordinals on parent body surface of interest 
      //
      Plato::OrdinalType tElementOrdinal = tSideSetLocalElemOrds(aSideOrdinal);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<mNumNodesPerFace; tNodeOrd++){
        tFaceLocalNodeOrds(tNodeOrd) = tSideSetLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tNodeOrd);
      }
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> 
        tGradient(ConfigScalarType(0.));
      tComputeGradient(tElementOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute state gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(tElementOrdinal,tStateWS,tGradient,tStateGradient);
      // interpolate temperature field from nodes to integration point
      //
      NodeStateScalarType tTemperature = 
        tInterpolateFromNodal(tElementOrdinal,tBasisValuesOnParentBodyElemSurface,tTempWS);
      // compute mechanical and thermal deformation gradients
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tMechDefGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tMechDefGradient);   
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> 
        tTempDefGradient(NodeStateScalarType(0.));
      tComputeThermalDeformationGradient(tTemperature,tTempDefGradient);
      // compute multiplicative decomposition for the thermo-mechanical deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
        tThermoMechDefGrad(ResultScalarType(0.));
      tComputeThermoElasticDeformationGradient(tTempDefGradient,tMechDefGradient,tThermoMechDefGrad);
      ResultScalarType tDetDefGrad = Plato::determinant(tThermoMechDefGrad);
      // invert transpose of deformation gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
        tDefGradientT = Plato::transpose(tThermoMechDefGrad);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
        tInverseDefGradientT = Plato::invert(tDefGradientT);
      // evaluate basis function gradients on parent face element
      //
      auto tCubPointOnParentFaceElem   = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // compute area weighted normal vector
      //
      Plato::Array<mNumSpatialDims,ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tElementOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );
      // evaluate follower pressure and save result to result workset
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerFace; tNode++){
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
          ResultScalarType tValue(0.0);
          auto tDofOrdinal = (tFaceLocalNodeOrds[tNode] * mNumSpatialDims) + tDimI + DofOffset;
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
            tValue += tInverseDefGradientT(tDimI,tDimJ) * ( tWeightedNormalVector(tDimJ) * tFlux(tDimJ) );
          }
          tValue =  tBasisValuesOnParentBodyElemSurface(tNode) * tDetDefGrad * tValue * 
            tCubWeightOnParentBodyElemSurface * aScale * tNormalMultiplier;
          Kokkos::atomic_add( &tResultWS(tElementOrdinal,tDofOrdinal), tValue );
        }
      }
    });
  }

private:
  /// @brief initialize member data
  /// @param aSubList 
  void 
  initialize(
    Teuchos::ParameterList & aSubList
  )
  {
    if(!aSubList.isParameter("Sides")){
      ANALYZE_THROWERR(std::string("ERROR: Input argument ('Sides') is not defined in Neumann boundary condition ") +   
        "parameter list, side sets for Neumann boundary conditions cannot be determined")
    }
    mSideSetName = aSubList.get<std::string>("Sides");
    if(!aSubList.isParameter("Material Model")){
      ANALYZE_THROWERR(std::string("ERROR: Input argument ('Material Model') is not defined in Neumann boundary ") +   
        "condition parameter list, side sets for Neumann boundary conditions cannot be determined")
    }
    mMaterialName = aSubList.get<std::string>("Material Model");
  }
};

} // namespace thermomechanical

} // namespace Plato