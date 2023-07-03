/*
 * ResidualElastostaticTotalLagrangian_decl.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "BodyLoads.hpp"
#include "SpatialModel.hpp"

#include "base/ResidualBase.hpp"
#include "bcs/neumann/NeumannBCs.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/mechanical/nonlinear/StressEvaluator.hpp"

namespace Plato
{
    
namespace Elliptic
{
  
/// @brief Evaluate nonlinear elastostatic residual of the form:
///  \f[
///      \int_{\Omega_0}\left( P_{ji}\delta{F}_{ij}-\rho_0 b_i \delta{u}_i \right)d\Omega_0 
///    - \int_{\Gamma_0}t_i^0\delta{u}_id\Gamma_0 = 0
///  \f]
/// A total Lagrangian formulation is used to represent the residual. \f$\Omega_0\f$ is the 
/// undeformed configuration, \f$\Gamma_0\f$ is the boundary on the undeformed configuration,
/// \f$P\f$ is the nominal stress, \f$F\f$ is the deformation gradient, \f$\rho_0\f$ is the
/// material density, \f$b\f$ are the body forces, \f$t^0\f$ are the traction forces, and 
/// \f$u\f$ are the displacements
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ResidualElastostaticTotalLagrangian : public Plato::ResidualBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell = ElementType::mNumDofsPerCell;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;
  /// @brief number of integration points per cell
  static constexpr auto mNumGaussPoints = ElementType::mNumGaussPoints;
  /// @brief local typename for base class
  using FunctionBaseType = Plato::ResidualBase;
  /// @brief contains mesh and model information
  using FunctionBaseType::mSpatialDomain;
  /// @brief output database
  using FunctionBaseType::mDataMap;
  /// @brief contains degrees of freedom names 
  using FunctionBaseType::mDofNames;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
  /// @brief stress evaluator
  std::shared_ptr<Plato::StressEvaluator<EvaluationType>> mStressEvaluator;
  /// @brief natural boundary conditions evaluator
  std::shared_ptr<Plato::NeumannBCs<EvaluationType>> mBoundaryForces;
  /// @brief body loads evaluator
  std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
  /// @brief output plot table, contains requested output quantities of interests
  std::vector<std::string> mPlotTable;

public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  /// @param [in] aParamList     input problem parameters
  ResidualElastostaticTotalLagrangian(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList
  );

  /// @brief class destructor
  ~ResidualElastostaticTotalLagrangian(){}

  /// @fn type
  /// @brief get residual type
  /// @return residual_t enum
  Plato::Elliptic::residual_t
  type() 
  const
  { return Plato::Elliptic::residual_t::NONLINEAR_MECHANICAL; }

  /// @fn postProcess
  /// @brief post process solution database before output
  /// @param [in] aSolutions solution database
  void
  postProcess(
    const Plato::Solutions & aSolutions
  );

  /// @fn evaluate
  /// @brief evaluate internal forces
  /// @param [in,out] aWorkSets domain and range workset database
  /// @param [in]     aCycle    scalar
  void
  evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) const;

  /// @fn evaluate_boundary
  /// @brief evaluate boundary forces
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        scalar
  void
  evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) const;

private:
  /// @fn initialize
  /// @brief initialize member data
  /// @param [in] aParamList input problem parameters
  void initialize(
    Teuchos::ParameterList & aParamList
  );
};
    
} // namespace Elliptic

} // namespace Plato
