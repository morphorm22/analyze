#ifndef PLATOTESTHELPERS_HPP_
#define PLATOTESTHELPERS_HPP_

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"

#include <BamG.hpp>

#include <Teuchos_RCP.hpp>

#include <string>
#include <vector>

namespace Plato {
namespace TestHelpers {
/******************************************************************************//**
 * \brief get view from device
 *
 * \param[in] aView data on device
 * @returns Mirror on host
**********************************************************************************/
template <typename ViewType> 
typename ViewType::HostMirror get(ViewType aView) {
  using RetType = typename ViewType::HostMirror;
  RetType tView = Kokkos::create_mirror(aView);
  Kokkos::deep_copy(tView, aView);
  return tView;
}

/******************************************************************************/
/*! Return a box (cube) along with the spec used to generate it.
/*! @sa get_box_mesh
*/
auto get_box_mesh_with_spec(
    const std::string& aMeshType, 
    Plato::OrdinalType aMeshIntervals,
    const std::string& aFileName = "BamG_unit_test_mesh.exo") 
    -> std::tuple<Plato::Mesh, BamG::MeshSpec>;

/******************************************************************************/
/*! Return a box (cube) mesh.
 * \param aMeshIntervals Number of mesh intervals through the thickness.
 * \param aMeshType Mesh type (i.e., TET4, hex8, TRI3, quad4, bar2, etc.)
 *
 * The mesh will have sidesets and nodesets on all faces, edges, and vertices
 * named 'x+' for the positive x face, 'x+y-' for the positive x negative y edge,
 * and 'x-y+z-' for the negative x positive y negative z vertex, etc.
 */
Plato::Mesh
 get_box_mesh(
    std::string        aMeshType,
    Plato::OrdinalType aMeshIntervals,
    std::string        aFileName = "BamG_unit_test_mesh.exo");

/******************************************************************************/
/*! Return a box (cube) mesh.
 * \param aMeshType Mesh type (i.e., TET4, hex8, TRI3, quad4, bar2, etc.)
 * \param aMeshIntervalsX Number of mesh intervals in X, etc.
 * \param aMeshWidthX Width of mesh in X, etc.
 *
 * The mesh will have sidesets and nodesets on all faces, edges, and vertices
 * named 'x+' for the positive x face, 'x+y-' for the positive x negative y edge,
 * and 'x-y+z-' for the negative x positive y negative z vertex, etc.
 */
Plato::Mesh
 get_box_mesh(
    std::string        aMeshType,
    Plato::Scalar      aMeshWidthX,
    Plato::OrdinalType aMeshIntervalsX,
    Plato::Scalar      aMeshWidthY=1.0,
    Plato::OrdinalType aMeshIntervalsY=1,
    Plato::Scalar      aMeshWidthZ=1.0,
    Plato::OrdinalType aMeshIntervalsZ=1);

/******************************************************************************//**
 * \brief Set Dirichlet boundary condition values for specified degree of freedom.
 *   Specialized for 2-D applications
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofValues  vector of Dirichlet boundary condition values
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 * \param [in] aSetValue   value to set
 *
 **********************************************************************************/
void set_dof_value_in_vector_on_boundary_2D(Plato::Mesh aMesh,
                                            const std::string & aBoundaryID,
                                            const Plato::ScalarVector & aDofValues,
                                            const Plato::OrdinalType & aDofStride,
                                            const Plato::OrdinalType & aDofToSet,
                                            const Plato::Scalar & aSetValue);

/******************************************************************************//**
 * \brief Set Dirichlet boundary condition values for specified degree of freedom.
 *   Specialized for 3-D applications.
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofValues  vector of Dirichlet boundary condition values
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 * \param [in] aSetValue   value to set
 *
 **********************************************************************************/
void set_dof_value_in_vector_on_boundary_3D(Plato::Mesh aMesh,
                                            const std::string & aBoundaryID,
                                            const Plato::ScalarVector & aDofValues,
                                            const Plato::OrdinalType & aDofStride,
                                            const Plato::OrdinalType & aDofToSet,
                                            const Plato::Scalar & aSetValue);

/******************************************************************************//**
 * \brief Return list of Dirichlet degree of freedom indices, specialized for 2-D applications.
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 *
 * \return list of Dirichlet indices
 *
 **********************************************************************************/
Plato::OrdinalVector
get_dirichlet_indices_on_boundary_2D(
          Plato::Mesh          aMesh,
    const std::string        & aBoundaryID,
    const Plato::OrdinalType & aDofStride,
    const Plato::OrdinalType & aDofToSet);

/******************************************************************************//**
 * \brief Return list of Dirichlet degree of freedom indices, specialized for 3-D applications.
 *
 * \param [in]     aMesh       finite element mesh
 * \param [in]     aBoundaryID boundary identifier
 * \param [in]     aDofStride  degree of freedom stride
 * \param [in]     aDofToSet   degree of freedom index to set
 *
 * \return list of Dirichlet indices
 *
 **********************************************************************************/
Plato::OrdinalVector
get_dirichlet_indices_on_boundary_3D(
          Plato::Mesh          aMesh,
    const std::string        & aBoundaryID,
    const Plato::OrdinalType & aDofStride,
    const Plato::OrdinalType & aDofToSet);

/******************************************************************************//**
 * \brief set value for this Dirichlet boundary condition index
 *
 * \param [in] aDofValues vector of Dirichlet boundary condition values
 * \param [in] aDofStride degree of freedom stride
 * \param [in] aDofToSet  degree of freedom index to set
 * \param [in] aSetValue  value to set
 *
 **********************************************************************************/
void set_dof_value_in_vector(const Plato::ScalarVector & aDofValues,
                             const Plato::OrdinalType & aDofStride,
                             const Plato::OrdinalType & aDofToSet,
                             const Plato::Scalar & aSetValue);

/******************************************************************************//**
 * \brief Expands compressed row sparse matrix to a full (non-sparse) representation
 **********************************************************************************/
std::vector<std::vector<Plato::Scalar>>
to_full( Teuchos::RCP<Plato::CrsMatrixType> aInMatrix );

} // namespace TestHelpers
} // namespace Plato

#endif /* PLATOTESTHELPERS_HPP_ */
