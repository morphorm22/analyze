/*
 * SurfaceLoadIntegral.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "SurfaceIntegralUtilities.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM and UNIFORM COMPONENT.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class SurfaceLoadIntegral
{
private:
    /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDims = SpatialDim;
    /*!< number of spatial dimensions on face */
    static constexpr auto mNumSpatialDimsOnFace = mNumSpatialDims - static_cast<Plato::OrdinalType>(1);

    const std::string mSideSetName; /*!< side set name */
    const Plato::Array<NumDofs> mFlux; /*!< force vector values */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mCubatureRule; /*!< integration rule */

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
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfaceLoadIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::SurfaceLoadIntegral
(const std::string & aSideSetName, const Plato::Array<NumDofs>& aFlux) :
    mSideSetName(aSideSetName),
    mFlux(aFlux),
    mCubatureRule()
{
}
// class SurfaceLoadIntegral::SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::operator() function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfaceLoadIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::operator()(
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
    auto tNumFaces = tElementOrds.size();

    Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
    Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
    Plato::ScalarArray3DT<ConfigScalarType> tJacobian("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

    const auto tNodesPerFace = mNumSpatialDims;

    auto tFlux = mFlux;
    auto tCubatureWeight = mCubatureRule.getCubWeight();
    auto tBasisFunctions = mCubatureRule.getBasisFunctions();
    if(std::isfinite(tCubatureWeight) == false)
    {
        ANALYZE_THROWERR("Natural Boundary Condition: A non-finite cubature weight was detected.")
    }
    auto tCubWeightTimesScale = aScale * tCubatureWeight;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal)
    {
      auto tElementOrdinal = tElementOrds(aSideOrdinal);

      Plato::OrdinalType tLocalNodeOrds[tNodesPerFace];
      for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<tNodesPerFace; tNodeOrd++)
      {
          tLocalNodeOrds[tNodeOrd] = tNodeOrds(aSideOrdinal*tNodesPerFace+tNodeOrd);
      }

      ConfigScalarType tSurfaceAreaTimesCubWeight(0.0);
      tCalculateSurfaceJacobians(tElementOrdinal, aSideOrdinal, tLocalNodeOrds, aConfig, tJacobian);
      tCalculateSurfaceArea(aSideOrdinal, tCubWeightTimesScale, tJacobian, tSurfaceAreaTimesCubWeight);

      // project into aResult workset
      for( Plato::OrdinalType tNode=0; tNode<tNodesPerFace; tNode++)
      {
          for( Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
          {
              auto tElementDofOrdinal = tLocalNodeOrds[tNode] * DofsPerNode + tDof + DofOffset;
              aResult(tElementOrdinal,tElementDofOrdinal) +=
                  tBasisFunctions(tNode) * tFlux[tDof] * tSurfaceAreaTimesCubWeight;
          }
      }
    }, "surface load integral");
}
// class SurfaceLoadIntegral::operator()

}
// namespace Plato
