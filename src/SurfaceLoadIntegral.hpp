/*
 * SurfaceLoadIntegral.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "SurfaceArea.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM and UNIFORM COMPONENT.
 *
 * \tparam ElementType  Element type (e.g., MechanicsElement<Tet10>)
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<
  typename ElementType,
  Plato::OrdinalType NumDofs=ElementType::mNumSpatialDims,
  Plato::OrdinalType DofsPerNode=NumDofs,
  Plato::OrdinalType DofOffset=0 >
class SurfaceLoadIntegral
{
private:
    /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;
    /*!< number of spatial dimensions on face */
    static constexpr auto mNumSpatialDimsOnFace = mNumSpatialDims - static_cast<Plato::OrdinalType>(1);

    static constexpr auto mNumNodesPerFace = ElementType::mNumNodesPerFace;

    const std::string mSideSetName; /*!< side set name */
    const Plato::Array<NumDofs> mFlux; /*!< force vector values */
//    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mCubatureRule; /*!< integration rule */

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    SurfaceLoadIntegral(const std::string & aSideSetName, const Plato::Array<NumDofs>& aFlux);

    /***************************************************************************//**
     * \brief Evaluate natural boundary condition surface integrals.
     *
     * \tparam StateScalarType   state forward automatically differentiated (FAD) type
     * \tparam ControlScalarType control FAD type
     * \tparam ConfigScalarType  configuration FAD type
     * \tparam ResultScalarType  result FAD type
     *
     * \param [in]  aSpatialModel Plato spatial model
     * \param [in]  aState        2-D view of state variables.
     * \param [in]  aControl      2-D view of control variables.
     * \param [in]  aConfig       3-D view of configuration variables.
     * \param [out] aResult       Assembled vector to which the boundary terms will be added
     * \param [in]  aScale        scalar multiplier
     *
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void operator()(
        const Plato::SpatialModel                          & aSpatialModel,
        const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
        const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
              Plato::Scalar aScale) const;
};
// class SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::SurfaceLoadIntegral constructor definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfaceLoadIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::SurfaceLoadIntegral
(const std::string & aSideSetName, const Plato::Array<NumDofs>& aFlux) :
    mSideSetName(aSideSetName),
    mFlux(aFlux)
{
}
// class SurfaceLoadIntegral::SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::operator() function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfaceLoadIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::operator()(
    const Plato::SpatialModel                          & aSpatialModel,
    const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
    const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
          Plato::Scalar aScale
) const
{
    auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    Plato::OrdinalType tNumFaces = tElementOrds.size();

    Plato::SurfaceArea<ElementType> surfaceArea;

    const auto tNodesPerFace = mNumNodesPerFace;

    auto tFlux = mFlux;
    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tCubaturePoints  = ElementType::Face::getCubPoints();
    auto tNumPoints = tCubatureWeights.size();

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
    LAMBDA_EXPRESSION(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tElementOrdinal = tElementOrds(aSideOrdinal);

      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
      for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<tNodesPerFace; tNodeOrd++)
      {
          tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*tNodesPerFace+tNodeOrd);
      }

      auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
      auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
      auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
      auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

      ResultScalarType tSurfaceArea(0.0);
      surfaceArea(aSideOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tSurfaceArea);
      tSurfaceArea *= aScale;
      tSurfaceArea *= tCubatureWeight;

      // project into aResult workset
      for( Plato::OrdinalType tNode=0; tNode<tNodesPerFace; tNode++)
      {
          for( Plato::OrdinalType tDof=0; tDof<mNumSpatialDims; tDof++)
          {
              auto tElementDofOrdinal = tLocalNodeOrds[tNode] * DofsPerNode + tDof + DofOffset;
              aResult(tElementOrdinal,tElementDofOrdinal) +=
                  tBasisValues(tNode) * tFlux[tDof] * tSurfaceArea;
          }
      }
    }, "surface load integral");
}
// class SurfaceLoadIntegral::operator()

}
// namespace Plato
