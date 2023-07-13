/*
 *  FollowerPressure.hpp
 *
 *  Created on: July 5, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "bcs/neumann/NeumannBoundaryConditionBase.hpp"

#include "GradientMatrix.hpp"
#include "utilities/WeightedNormalVector.hpp"
#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"

namespace Plato
{

namespace mechanical
{

/// @class FollowerPressure
/// @brief evaluates mechanical follower pressure:
/// \[
///  p_{f}=-\int_{\Gamma^{0}_{\bar{p}}}\delta{u} J F_{ij}^{-T}(n_j \bar{p})\ d\Gamma^0,\quad{i,j}=1,\dots,N_{dim}
/// \]
/// where \f$p_f\f$ denotes the follower pressure, \f$\Gamma^0\f$ is the undeformed configuration,
/// \f$\Gamma^{0}_{\bar{p}}\f$ is the loaded surface on the undeformed configuration, \f$J=\det(F)\f$, 
/// \f$F_{ij}\f$ is the deformation gradient, and \f$\bar{p}\f$ denotes the pressure.
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
  /// @brief topological element typename
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
  /// @brief number of integration points per face
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using StrainScalarType = typename Plato::fad_type_t<BodyElementBase, StateScalarType, ConfigScalarType>;
  /// @brief set local typename for base class
  using BaseClassType = Plato::NeumannBoundaryConditionBase<NumForceDof>;
  /// @brief flux magnitude
  using BaseClassType::mFlux;
  /// @brief side set name 
  using BaseClassType::mSideSetName;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  /// @param [in] aSubList   neumann boundary condition parameter list
  FollowerPressure(
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

  /// @fn flux
  /// @brief update flux vector values
  /// @param [in] aFlux flux vector
  void 
  flux(
    const Plato::Array<NumForceDof> & aFlux
  )
  { mFlux = aFlux; }

  /// @brief evaluate mechanical follower pressure
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
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
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
    // pressure acts towards the surface; therefore, -1.0 is used to invert the outward facing normal inwards.
    //
    auto tFlux = mFlux;
    Plato::Scalar tNormalMultiplier(-1.0);
    Plato::OrdinalType tNumCellsOnSideSet = tSideSetLocalElemOrds.size();
    Kokkos::parallel_for("follower pressure",
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
      // compute interpolation function gradient
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> 
        tGradient(ConfigScalarType(0.));
      tComputeGradient(tElementOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute state gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(tElementOrdinal,tStateWS,tGradient,tStateGradient);
      // compute deformation gradient (F) and compute its determinant (det(F))
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDefGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);    
      StrainScalarType tDetDefGrad = Plato::determinant(tDefGradient);
      // invert transpose of deformation gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDefGradientT = Plato::transpose(tDefGradient);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tInverseDefGradientT = Plato::invert(tDefGradientT);
      // evaluate basis function gradients on parent face element
      //
      auto tCubPointOnParentFaceElem   = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // compute area weighted normal vector, i.e., surface area is already applied to normal vector
      //
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
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
          tValue = tBasisValuesOnParentBodyElemSurface(tNode) * tDetDefGrad * tValue *
            tCubWeightOnParentBodyElemSurface * aScale * tNormalMultiplier;
          Kokkos::atomic_add( &tResultWS(tElementOrdinal,tDofOrdinal), tValue );
        }
      }
    });
  }
};

} // namespace mechanical

} // namespace Plato