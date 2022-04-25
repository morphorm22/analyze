
/*!
 * Plato_MeshMapUtils.hpp
 *
 * Created on: Oct 1, 2020
 *
 */

#ifndef PLATO_MESHMAP_UTILS_HPP_
#define PLATO_MESHMAP_UTILS_HPP_

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include "ElementBase.hpp"
#include "SpatialModel.hpp"

namespace Plato {
namespace Geometry {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;

struct BoundingBoxes
{
  double *d_x0;
  double *d_y0;
  double *d_z0;
  double *d_x1;
  double *d_y1;
  double *d_z1;
  int N;
};

struct Spheres
{
  double *d_x;
  double *d_y;
  double *d_z;
  double *d_r;
  int N;
};

struct Points
{
  double *d_x;
  double *d_y;
  double *d_z;
  int N;
};

} // namespace Geometry
} // namespace Plato


namespace ArborX
{
namespace Traits
{
template <>
struct Access<Plato::Geometry::BoundingBoxes, PrimitivesTag>
{
  inline static std::size_t size(Plato::Geometry::BoundingBoxes const &boxes) { return boxes.N; }
  KOKKOS_INLINE_FUNCTION static Box get(Plato::Geometry::BoundingBoxes const &boxes, std::size_t i)
  {
    return {{boxes.d_x0[i], boxes.d_y0[i], boxes.d_z0[i]},
            {boxes.d_x1[i], boxes.d_y1[i], boxes.d_z1[i]}};
  }
  using memory_space = Plato::Geometry::MemSpace;
};

template <>
struct Access<Plato::Geometry::Points, PrimitivesTag>
{
  inline static std::size_t size(Plato::Geometry::Points const &points) { return points.N; }
  KOKKOS_INLINE_FUNCTION static Point get(Plato::Geometry::Points const &points, std::size_t i)
  {
    return {{points.d_x[i], points.d_y[i], points.d_z[i]}};
  }
  using memory_space = Plato::Geometry::MemSpace;
};

template <>
struct Access<Plato::Geometry::Spheres, PredicatesTag>
{
  inline static std::size_t size(Plato::Geometry::Spheres const &d) { return d.N; }
  KOKKOS_INLINE_FUNCTION static auto get(Plato::Geometry::Spheres const &d, std::size_t i)
  {
    return intersects(Sphere{{{d.d_x[i], d.d_y[i], d.d_z[i]}}, d.d_r[i]});
  }
  using memory_space = Plato::Geometry::MemSpace;
};

template <>
struct Access<Plato::Geometry::Points, PredicatesTag>
{
  inline static std::size_t size(Plato::Geometry::Points const &d) { return d.N; }
  KOKKOS_INLINE_FUNCTION static auto get(Plato::Geometry::Points const &d, std::size_t i)
  {
    return intersects(Point{d.d_x[i], d.d_y[i], d.d_z[i]});
  }
  using memory_space = Plato::Geometry::MemSpace;
};

} // namespace Traits
} // namespace ArborX


namespace Plato {
namespace Geometry {

enum Dim { X=0, Y, Z };

/***************************************************************************//**
* @brief Functor that computes position in local coordinates of a point given
         in global coordinates then returns the basis values at that local
         point.

  The local position is computed as follows.  Given:
  \f{eqnarray*}{
    \bar{x}^h(\xi) = N_I(\xi) \bar{x}_I \\
    N_I = \left\{\begin{array}{cccc}
              x_l & y_l & z_l & 1-x_l-y_l-z_l
           \end{array}\right\}^T
  \f}
  Find: \f$ x_l \f$, \f$ y_l \f$, and \f$ z_l \f$.

  Simplifying the above yields:
  \f[
    \left[\begin{array}{ccc}
      x_1-x_4 & x_2-x_4 & x_3-x_4 \\
      y_1-y_4 & y_2-y_4 & y_3-y_4 \\
      z_1-z_4 & z_2-z_4 & z_3-z_4 \\
    \end{array}\right]
    \left\{\begin{array}{c}
      x_l \\ y_l \\ z_l
    \end{array}\right\} =
    \left\{\begin{array}{c}
      x^h-x_4 \\ y^h-y_4 \\ z^h-z_4
    \end{array}\right\}
  \f]
  Below directly solves the linear system above for \f$x_l\f$, \f$ y_l \f$, and
  \f$ z_l \f$ then evaluates the basis.
*******************************************************************************/
template <typename ElementT, typename ScalarT>
struct GetBasis
{
    using ScalarArrayT  = typename Plato::ScalarVectorT<ScalarT>;
    using VectorArrayT  = typename Plato::ScalarMultiVectorT<ScalarT>;
    using OrdinalT      = typename ScalarArrayT::size_type;
    using OrdinalArrayT = typename Plato::ScalarVectorT<OrdinalT>;

    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2Nodes;
    const Plato::ScalarVectorT<const Plato::Scalar> mCoords;

    GetBasis(Plato::Mesh aMesh) :
      mCells2Nodes(aMesh->Connectivity()),
      mCoords(aMesh->Coordinates()) {}

    /******************************************************************************//**
     * @brief Get node locations from global coordinates
     * @param [in]  indices of nodes comprised by the element
     * @param [out] node locations
    **********************************************************************************/
    DEVICE_TYPE inline
    Plato::Matrix<ElementT::mNumNodesPerCell, ElementT::mNumSpatialDims>
    getNodeLocations(
      const Plato::Array<ElementT::mNumNodesPerCell, Plato::OrdinalType> & aNodeOrdinals
    ) const
    {
        Plato::Matrix<ElementT::mNumNodesPerCell, ElementT::mNumSpatialDims> tNodeLocations;
        for(Plato::OrdinalType iNode=0; iNode<ElementT::mNumNodesPerCell; iNode++)
        {
            for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
            {
                tNodeLocations(iNode, iDim) = mCoords(aNodeOrdinals(iNode)*ElementT::mNumSpatialDims+iDim);
            }
        }
        return tNodeLocations;
    }

    /******************************************************************************//**
     * @brief Find local coordinates from global coordinates, compute basis values
     * @param [in]  position in global coordinates
     * @param [in]  indices of nodes comprised by the element
     * @param [out] basis values
    **********************************************************************************/
    DEVICE_TYPE inline void
    basis(
      const Plato::Array<ElementT::mNumSpatialDims, ScalarT>             & aPhysicalLocation,
      const Plato::Array<ElementT::mNumNodesPerCell, Plato::OrdinalType> & aNodeOrdinals,
            Plato::Array<ElementT::mNumNodesPerCell, ScalarT>            & aBases
    ) const
    {
        Plato::Array<ElementT::mNumSpatialDims, ScalarT> tParentCoords(0.0);

        Plato::Matrix<ElementT::mNumNodesPerCell, ElementT::mNumSpatialDims, ScalarT>
        tNodeLocations = getNodeLocations(aNodeOrdinals);

        aBases = ElementT::basisValues(tParentCoords);

        // compute current difference 
        Plato::Array<ElementT::mNumSpatialDims, ScalarT> tDiff(0.0);
        for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
        {
            ScalarT tPhysical(0.0);
            for(Plato::OrdinalType iNode=0; iNode<ElementT::mNumNodesPerCell; iNode++)
            {
                tPhysical += aBases(iNode)*tNodeLocations(iNode,iDim);
            }
            tDiff(iDim) = aPhysicalLocation(iDim) - tPhysical;
        }

        ScalarT tError(0.0);
        for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
        {
            tError += tDiff(iDim)*tDiff(iDim);
        }
      
        Plato::OrdinalType tIteration = 0;
        while(tError > 1e-3 && tIteration < 4)
        {
            auto tJacobian = Plato::ElementBase<ElementT>::template jacobian<ScalarT>(tParentCoords, tNodeLocations);
            auto tJacInv = Plato::invert(tJacobian);

            // multiply tJacInv by tDiff and subtract from tParentCoords
            for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
            {
                for(Plato::OrdinalType jDim=0; jDim<ElementT::mNumSpatialDims; jDim++)
                {
                    tParentCoords(iDim) += tJacInv(iDim, jDim) * tDiff(jDim);
                }
            }
        
            aBases = ElementT::basisValues(tParentCoords);

            // compute current difference 
            Plato::Array<ElementT::mNumSpatialDims, ScalarT> tDiff(0.0);
            for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
            {
                ScalarT tPhysical(0.0);
                for(Plato::OrdinalType iNode=0; iNode<ElementT::mNumNodesPerCell; iNode++)
                {
                    tPhysical += aBases(iNode)*tNodeLocations(iNode,iDim);
                }
                tDiff(iDim) = aPhysicalLocation(iDim) - tPhysical;
            }

            tError = 0.0;
            for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
            {
                tError += tDiff(iDim)*tDiff(iDim);
            }
            tIteration++;
        }
    }


    /******************************************************************************//**
     * @brief Find local coordinates from global coordinates, compute basis values, and
              assembles them into the columnMap and entries of a sparse matrix.
     * @param [in]  aLocations view of points (D, N)
     * @param [in]  aNodeOrdinal index of point for which to determine local coords
     * @param [in]  aElemOrdinal index of element whose bases will be used for interpolation
     * @param [in]  aEntryOrdinal index into aColumnMap and aEntries
     * @param [out] aColumnMap of the sparse matrix
     * @param [out] aEntries of the sparse matrix
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(
      VectorArrayT  aLocations,
      OrdinalT      aNodeOrdinal,
      int           aElemOrdinal,
      OrdinalT      aEntryOrdinal,
      OrdinalArrayT aColumnMap,
      ScalarArrayT  aEntries) const
    {
        // get input point values
        Plato::Array<ElementT::mNumSpatialDims, ScalarT> tPhysicalPoint;
        for(Plato::OrdinalType iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
        {
            tPhysicalPoint(iDim) = aLocations(iDim, aNodeOrdinal);
        }

        // get node indices
        Plato::Array<ElementT::mNumNodesPerCell, Plato::OrdinalType> tNodeOrdinals;
        for(Plato::OrdinalType iOrd=0; iOrd<ElementT::mNumNodesPerCell; iOrd++)
        {
            tNodeOrdinals(iOrd) = mCells2Nodes(aElemOrdinal*ElementT::mNumNodesPerCell+iOrd);
        }

        Plato::Array<ElementT::mNumNodesPerCell, ScalarT> tBases;

        basis(tPhysicalPoint, tNodeOrdinals, tBases);

        for(Plato::OrdinalType iOrd=0; iOrd<ElementT::mNumNodesPerCell; iOrd++)
        {
            aColumnMap(aEntryOrdinal+iOrd) = tNodeOrdinals(iOrd);
            aEntries(aEntryOrdinal+iOrd)   = tBases(iOrd);
        }
    }

#ifdef NOPE // delete this
    /******************************************************************************//**
     * @brief Find local coordinates from global coordinates and compute basis values
     * @param [in]  aLocations view of points (D, N)
     * @param [in]  aNodeOrdinal index of point for which to determine local coords
     * @param [in]  aElemOrdinal index of element whose bases will be used for interpolation
     * @param [out] aBases basis values
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(
      VectorArrayT  aLocations,
      OrdinalT      aNodeOrdinal,
      int           aElemOrdinal,
      ScalarT       aBases[cNVertsPerElem]) const
    {
        // get input point values
        ScalarT Xh=aLocations(Dim::X,aNodeOrdinal),
                Yh=aLocations(Dim::Y,aNodeOrdinal),
                Zh=aLocations(Dim::Z,aNodeOrdinal);

        // get vertex indices
        OrdinalT i0 = mCells2Nodes[aElemOrdinal*cNVertsPerElem  ];
        OrdinalT i1 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+1];
        OrdinalT i2 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+2];
        OrdinalT i3 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+3];

        ScalarT b0, b1, b2, b3;

        basis(Xh, Yh, Zh,
              i0, i1, i2, i3,
              b0, b1, b2, b3);

        aBases[0] = b0;
        aBases[1] = b1;
        aBases[2] = b2;
        aBases[3] = b3;
    }
#endif

    /******************************************************************************//**
     * @brief Find local coordinates from global coordinates and compute basis values
     * @param [in]  aElemOrdinal index of element whose bases will be used for interpolation
     * @param [in]  aLocation of point (D)
     * @param [out] aBases basis values
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(
      Plato::OrdinalType                                  aElemOrdinal,
      Plato::Array<ElementT::mNumSpatialDims, ScalarT>    aPhysicalLocation,
      Plato::Array<ElementT::mNumNodesPerCell, ScalarT> & aBases
    ) const
    {
        // get node indices
        Plato::Array<ElementT::mNumNodesPerCell, Plato::OrdinalType> tNodeOrdinals;
        for(Plato::OrdinalType iOrd=0; iOrd<ElementT::mNumNodesPerCell; iOrd++)
        {
            tNodeOrdinals(iOrd) = mCells2Nodes(aElemOrdinal*ElementT::mNumNodesPerCell+iOrd);
        }

        basis(aPhysicalLocation, tNodeOrdinals, aBases);
    }
};

template<Plato::OrdinalType NumSpatialDims>
void inline
search(
    Plato::ScalarMultiVector         aMin,
    Plato::ScalarMultiVector         aMax,
    Plato::ScalarMultiVector         aMappedLocations,
    Kokkos::View<int*, DeviceType> & aIndices,
    Kokkos::View<int*, DeviceType> & aOffset
);

template<>
void inline
search<2>(
    Plato::ScalarMultiVector         aMin,
    Plato::ScalarMultiVector         aMax,
    Plato::ScalarMultiVector         aMappedLocations,
    Kokkos::View<int*, DeviceType> & aIndices,
    Kokkos::View<int*, DeviceType> & aOffset
) {
    Plato::OrdinalType tNumElements = aMin.extent(1);
    Plato::OrdinalType tNumLocations = aMappedLocations.extent(1);

    auto d_x0 = Kokkos::subview(aMin, (size_t)Dim::X, Kokkos::ALL());
    auto d_y0 = Kokkos::subview(aMin, (size_t)Dim::Y, Kokkos::ALL());
    Plato::ScalarVector d_z0("zeros", tNumElements);

    auto d_x1 = Kokkos::subview(aMax, (size_t)Dim::X, Kokkos::ALL());
    auto d_y1 = Kokkos::subview(aMax, (size_t)Dim::Y, Kokkos::ALL());
    Plato::ScalarVector d_z1("zeros", tNumElements);

    // construct search tree
    ArborX::BVH<DeviceType>
      bvh{BoundingBoxes{d_x0.data(), d_y0.data(), d_z0.data(),
                        d_x1.data(), d_y1.data(), d_z1.data(), tNumElements}};

    // conduct search for bounding box elements
    auto d_x = Kokkos::subview(aMappedLocations, (size_t)Dim::X, Kokkos::ALL());
    auto d_y = Kokkos::subview(aMappedLocations, (size_t)Dim::Y, Kokkos::ALL());
    Plato::ScalarVector d_z("zeros", tNumLocations);

    bvh.query(Points{d_x.data(), d_y.data(), d_z.data(), static_cast<int>(tNumLocations)}, aIndices, aOffset);
}


template<>
void inline
search<3>(
    Plato::ScalarMultiVector         aMin,
    Plato::ScalarMultiVector         aMax,
    Plato::ScalarMultiVector         aMappedLocations,
    Kokkos::View<int*, DeviceType> & aIndices,
    Kokkos::View<int*, DeviceType> & aOffset
) {
    Plato::OrdinalType tNumElements = aMin.extent(1);
    Plato::OrdinalType tNumLocations = aMappedLocations.extent(1);

    auto d_x0 = Kokkos::subview(aMin, (size_t)Dim::X, Kokkos::ALL());
    auto d_y0 = Kokkos::subview(aMin, (size_t)Dim::Y, Kokkos::ALL());
    auto d_z0 = Kokkos::subview(aMin, (size_t)Dim::Z, Kokkos::ALL());

    auto d_x1 = Kokkos::subview(aMax, (size_t)Dim::X, Kokkos::ALL());
    auto d_y1 = Kokkos::subview(aMax, (size_t)Dim::Y, Kokkos::ALL());
    auto d_z1 = Kokkos::subview(aMax, (size_t)Dim::Z, Kokkos::ALL());

    // construct search tree
    ArborX::BVH<DeviceType>
      bvh{BoundingBoxes{d_x0.data(), d_y0.data(), d_z0.data(),
                        d_x1.data(), d_y1.data(), d_z1.data(), tNumElements}};

    // conduct search for bounding box elements
    auto d_x = Kokkos::subview(aMappedLocations, (size_t)Dim::X, Kokkos::ALL());
    auto d_y = Kokkos::subview(aMappedLocations, (size_t)Dim::Y, Kokkos::ALL());
    auto d_z = Kokkos::subview(aMappedLocations, (size_t)Dim::Z, Kokkos::ALL());

    bvh.query(Points{d_x.data(), d_y.data(), d_z.data(), static_cast<int>(tNumLocations)}, aIndices, aOffset);
}


/***************************************************************************//**
* @brief Find element that contains each mapped node
 * @param [in]  aLocations location of mesh nodes
 * @param [in]  aMappedLocations mapped location of mesh nodes
 * @param [out] aParentElements if node is mapped, index of parent element.

   If a node is mapped (i.e., aLocations(*,node_id)!=aMappedLocations(*,node_id))
   and the parent element is found, aParentElements(node_id) is set to the index
   of the parent element.
   If a node is mapped but the parent element isn't found, aParentElements(node_id)
   is set to -2.
   If a node is not mapped, aParentElements(node_id) is set to -1.
*******************************************************************************/
template <typename ElementT, typename ScalarT>
void
findParentElements(
  Plato::Mesh aMesh,
  Plato::ScalarMultiVectorT<ScalarT> aLocations,
  Plato::ScalarMultiVectorT<ScalarT> aMappedLocations,
  Plato::ScalarVectorT<int> aParentElements,
  ScalarT aSearchTolerance)
{
    using OrdinalT = typename Plato::ScalarVectorT<ScalarT>::size_type;

    auto tNElems = aMesh->NumElements();
    Plato::ScalarMultiVectorT<ScalarT> tMin("min", ElementT::mNumSpatialDims, tNElems);
    Plato::ScalarMultiVectorT<ScalarT> tMax("max", ElementT::mNumSpatialDims, tNElems);

    // fill d_* data
    auto tCoords = aMesh->Coordinates();
    auto tCells2Nodes = aMesh->Connectivity();
    Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNElems), KOKKOS_LAMBDA(OrdinalT iCellOrdinal)
    {
        // set min and max of element bounding box to first node
        for(size_t iDim=0; iDim<ElementT::mNumSpatialDims; ++iDim)
        {
            OrdinalT tVertIndex = tCells2Nodes[iCellOrdinal*ElementT::mNumNodesPerCell];
            tMin(iDim, iCellOrdinal) = tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim];
            tMax(iDim, iCellOrdinal) = tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim];
        }
        // loop on remaining nodes to find min
        for(OrdinalT iVert=1; iVert<ElementT::mNumNodesPerCell; ++iVert)
        {
            OrdinalT tVertIndex = tCells2Nodes[iCellOrdinal*ElementT::mNumNodesPerCell + iVert];
            for(size_t iDim=0; iDim<ElementT::mNumSpatialDims; ++iDim)
            {
                if( tMin(iDim, iCellOrdinal) > tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim] )
                {
                    tMin(iDim, iCellOrdinal) = tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim];
                }
                else
                if( tMax(iDim, iCellOrdinal) < tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim] )
                {
                    tMax(iDim, iCellOrdinal) = tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim];
                }
            }
        }
        for(size_t iDim=0; iDim<ElementT::mNumSpatialDims; ++iDim)
        {
            ScalarT tLen = tMax(iDim, iCellOrdinal) - tMin(iDim, iCellOrdinal);
            tMax(iDim, iCellOrdinal) += aSearchTolerance * tLen;
            tMin(iDim, iCellOrdinal) -= aSearchTolerance * tLen;
        }
    }, "element bounding boxes");

    Kokkos::View<int*, DeviceType> tIndices("indices", 0), tOffset("offset", 0);
    search<ElementT::mNumSpatialDims>(tMin, tMax, aMappedLocations, tIndices, tOffset);

    // loop over indices and find containing element
    GetBasis<ElementT, ScalarT> tGetBasis(aMesh);
    auto tNumLocations = aParentElements.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumLocations), KOKKOS_LAMBDA(OrdinalT iNodeOrdinal)
    {
        Plato::Array<ElementT::mNumNodesPerCell, Plato::Scalar> tBasis(0.0);
        Plato::Array<ElementT::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);

        aParentElements(iNodeOrdinal) = -1;

        bool tMapped = false;
        for(OrdinalT iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
        {
            tMapped = tMapped || ( aLocations(iDim, iNodeOrdinal) != aMappedLocations(iDim, iNodeOrdinal) );
        }
        if( tMapped )
        {
            aParentElements(iNodeOrdinal) = -2;
            constexpr ScalarT cNotFound = -1e8; // big negative number ensures max min is found
            constexpr ScalarT cEpsilon = -1e-8; // small negative number for checking if float greater than 0
            ScalarT tMaxMin = cNotFound;
            OrdinalT tRunningNegCount = 4;
            typename Plato::ScalarVectorT<int>::value_type iParent = -2;
            for( int iElem=tOffset(iNodeOrdinal); iElem<tOffset(iNodeOrdinal+1); iElem++ )
            {
                auto tElemIndex = tIndices(iElem);
                for(OrdinalT iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
                {
                    tInPoint(iDim) = aMappedLocations(iDim, iNodeOrdinal);
                }

                tGetBasis(tElemIndex, tInPoint, tBasis);

                ScalarT tEleMin = tBasis[0];
                OrdinalT tNegCount = 0;
                for(OrdinalT iB=0; iB<ElementT::C1::mNumNodesPerCell; iB++)
                {
                    if( tBasis[iB] < tEleMin ) tEleMin = tBasis[iB];
                    if( tBasis[iB] < cEpsilon ) tNegCount += 1;
                }
                if( tNegCount < tRunningNegCount )
                {
                     tRunningNegCount = tNegCount;
                     tMaxMin = tEleMin;
                     iParent = tElemIndex;
                }
                else if ( ( tNegCount == tRunningNegCount ) && ( tEleMin > tMaxMin ) )
                {
                     tMaxMin = tEleMin;
                     iParent = tElemIndex;
                }
            }
            if( tMaxMin >= cEpsilon )
            {
                aParentElements(iNodeOrdinal) = iParent;
            }
            else
            {
                OrdinalT tBoundCheck = 0;
                for(OrdinalT iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
                {
                    ScalarT tBoundTol = aSearchTolerance * (tMax(iDim, iParent) - tMin(iDim, iParent));
                    if( tMaxMin < -tBoundTol ) tBoundCheck += 1;
                }
                if( tBoundCheck < 1 )
                {
                    aParentElements(iNodeOrdinal) = iParent;
                }
            }
        }
    }, "find parent element");
}
/***************************************************************************//**
* @brief Find element that contains each mapped node
 * @param [in]  aDomainCellMap map of local parent domain cell IDs to global cell IDs
 * @param [in]  aLocations location of mesh nodes
 * @param [in]  aMappedLocations mapped location of mesh nodes
 * @param [out] aParentElements if node is mapped, index of parent element.

   If a node is mapped (i.e., aLocations(*,node_id)!=aMappedLocations(*,node_id))
   and the parent element is found, aParentElements(node_id) is set to the index
   of the parent element.
   If a node is mapped but the parent element isn't found, aParentElements(node_id)
   is set to -2.
*******************************************************************************/
template <typename ElementT, typename ScalarT>
void
findParentElements(
        Plato::Mesh                          aMesh,
  const Plato::ScalarVectorT<int>          & aDomainCellMap,
        Plato::ScalarMultiVectorT<ScalarT>   aLocations,
        Plato::ScalarMultiVectorT<ScalarT>   aMappedLocations,
        Plato::ScalarVectorT<int>            aParentElements
)
{
    using OrdinalT = typename Plato::ScalarVectorT<ScalarT>::size_type;

    int tNElems = aDomainCellMap.size();
    Plato::ScalarMultiVectorT<ScalarT> tMin("min", ElementT::mNumSpatialDims, tNElems);
    Plato::ScalarMultiVectorT<ScalarT> tMax("max", ElementT::mNumSpatialDims, tNElems);

    constexpr ScalarT cRelativeTol = 1e-2;

    // fill d_* data
    auto tCoords = aMesh->Coordinates();
    auto tCells2Nodes = aMesh->Connectivity();
    auto tDomainCellMap = aDomainCellMap;

    Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNElems), KOKKOS_LAMBDA(OrdinalT iCellOrdinal)
    {
        OrdinalT tCellOrdinal = tDomainCellMap(iCellOrdinal);

        // set min and max of element bounding box to first node
        for(size_t iDim=0; iDim<ElementT::mNumSpatialDims; ++iDim)
        {
            OrdinalT tVertIndex = tCells2Nodes[tCellOrdinal*ElementT::mNumNodesPerCell];
            tMin(iDim, iCellOrdinal) = tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim];
            tMax(iDim, iCellOrdinal) = tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim];
        }
        // loop on remaining nodes to find min
        for(OrdinalT iVert=1; iVert<ElementT::mNumNodesPerCell; ++iVert)
        {
            OrdinalT tVertIndex = tCells2Nodes[tCellOrdinal*ElementT::mNumNodesPerCell + iVert];
            for(size_t iDim=0; iDim<ElementT::mNumSpatialDims; ++iDim)
            {
                if( tMin(iDim, iCellOrdinal) > tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim] )
                {
                    tMin(iDim, iCellOrdinal) = tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim];
                }
                else
                if( tMax(iDim, iCellOrdinal) < tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim] )
                {
                    tMax(iDim, iCellOrdinal) = tCoords[tVertIndex*ElementT::mNumSpatialDims+iDim];
                }
            }
        }
        for(size_t iDim=0; iDim<ElementT::mNumSpatialDims; ++iDim)
        {
            ScalarT tLen = tMax(iDim, iCellOrdinal) - tMin(iDim, iCellOrdinal);
            tMax(iDim, iCellOrdinal) += cRelativeTol * tLen;
            tMin(iDim, iCellOrdinal) -= cRelativeTol * tLen;
        }
    }, "element bounding boxes");

    Kokkos::View<int*, DeviceType> tIndices("indices", 0), tOffset("offset", 0);
    search<ElementT::mNumSpatialDims>(tMin, tMax, aMappedLocations, tIndices, tOffset);

    // loop over indices and find containing element
    GetBasis<ElementT, ScalarT> tGetBasis(aMesh);
    auto tNumLocations = aParentElements.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumLocations), KOKKOS_LAMBDA(OrdinalT iNodeOrdinal)
    {
        Plato::Array<ElementT::mNumNodesPerCell, Plato::Scalar> tBasis(0.0);
        Plato::Array<ElementT::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);

        aParentElements(iNodeOrdinal) = -2;
        constexpr ScalarT cNotFound = -1e8; // big negative number ensures max min is found
        constexpr ScalarT cEpsilon = -1e-8; // small negative number for checking if float greater than 0
        ScalarT tMaxMin = cNotFound;
        OrdinalT tRunningNegCount = 4;
        typename Plato::ScalarVectorT<int>::value_type iParent = -2;
        for( int iElem=tOffset(iNodeOrdinal); iElem<tOffset(iNodeOrdinal+1); iElem++ )
        {
            auto tElemIndex = tDomainCellMap(tIndices(iElem));

            for(OrdinalT iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
            {
                tInPoint(iDim) = aMappedLocations(iDim, iNodeOrdinal);
            }

            tGetBasis(tElemIndex, tInPoint, tBasis);

            ScalarT tEleMin = tBasis[0];
            OrdinalT tNegCount = 0;
            for(OrdinalT iB=0; iB<ElementT::mNumNodesPerCell; iB++)
            {
                if( tBasis[iB] < tEleMin ) tEleMin = tBasis[iB];
                if( tBasis[iB] < cEpsilon ) tNegCount += 1;
            }
            if( tNegCount < tRunningNegCount )
            {
                 tRunningNegCount = tNegCount;
                 tMaxMin = tEleMin;
                 iParent = tElemIndex;
            }
            else if ( ( tNegCount == tRunningNegCount ) && ( tEleMin > tMaxMin ) )
            {
                 tMaxMin = tEleMin;
                 iParent = tElemIndex;

            }
        }
        if( tMaxMin >= cEpsilon )
        {
            aParentElements(iNodeOrdinal) = iParent;
        }
        else
        {
            OrdinalT tBoundCheck = 0;
            for(OrdinalT iDim=0; iDim<ElementT::mNumSpatialDims; iDim++)
            {
                ScalarT tBoundTol = cRelativeTol * (tMax(iDim, iParent) - tMin(iDim, iParent));
                if( tMaxMin < -tBoundTol ) tBoundCheck += 1;
            }
            if( tBoundCheck < 1 )
            {
                aParentElements(iNodeOrdinal) = iParent;
            }
        }
    }, "find parent element");
}

}  // end namespace Geometry
}  // end namespace Plato

#endif
