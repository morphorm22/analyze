/*
 *  FollowerPressure.hpp
 *
 *  Created on: July 5, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "bcs/neumann/NeumannBoundaryConditionBase.hpp"

#include "GradientMatrix.hpp"
#include "WeightedNormalVector.hpp"
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
    /*
    // unpack worksets
    //
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    // local functors
    //
    Plato::SurfaceArea<BodyElementBase> tComputeFaceArea;
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeNormalVector;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    // get side set information
    //
    auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tNumFaces    = tElementOrds.size();
    auto tNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    auto tFaceOrds    = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    // get integration rule
    //
    auto tFlux = mFlux;
    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tCubaturePoints  = ElementType::Face::getCubPoints();
    auto tNumPoints = tCubatureWeights.size();
    // pressure acts towards the surface; therefore, -1.0
    // is used to invert the outward facing normal inwards.
    Plato::Scalar tNormalMultiplier(-1.0);
    Kokkos::parallel_for("surface integral",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      Plato::OrdinalType tElementOrdinal = tElementOrds(aSideOrdinal);
      Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
      for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
      {
        tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
      }
      // get cubature rule
      //
      auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
      auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
      auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
      auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);
      // compute interpolation function gradient
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> 
        tGradient(ConfigScalarType(0.));
      tComputeGradient(tElementOrdinal,tCubaturePoint,tConfigWS,tGradient,tVolume);
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
      // compute area weighted normal vector
      //
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVec;
      tWeightedNormalVector(tElementOrdinal, tLocalNodeOrds, tBasisGrads, tConfigWS, tWeightedNormalVec);
      // project into result workset
      //
      for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
      {
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
        {
          ResultScalarType tValue(0.0);
          auto tDofOrdinal = (tLocalNodeOrds[tNode] * mNumSpatialDims) + tDimI + DofOffset;
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
          {
            tValue += tInverseDefGradientT(tDimI,tDimJ) * ( tWeightedNormalVec(tDimJ) * tFlux(tDimJ) );
          }
          tValue =  tBasisValues(tNode) * tDetDefGrad * tValue * tCubatureWeight * aScale * tNormalMultiplier;
          Kokkos::atomic_add( &tResultWS(tElementOrdinal,tDofOrdinal), tValue );
        }
      }
    });
  */
  }
};

} // namespace mechanical

} // namespace Plato