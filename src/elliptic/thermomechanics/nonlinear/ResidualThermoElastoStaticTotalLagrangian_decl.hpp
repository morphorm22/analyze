/*
 * ResidualThermoElastoStaticTotalLagrangian_decl.hpp
 *
 *  Created on: June 17, 2023
 */

#pragma once

#include "Solutions.hpp"
#include "BodyLoads.hpp"

#include "base/WorksetBase.hpp"
#include "base/ResidualBase.hpp"

#include "bcs/neumann/NeumannBCs.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/mechanical/nonlinear/StressEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

/// @brief get cell second piola-kirchhoff stress tensor at integration point
/// @tparam ScalarType     data scalar type
/// @tparam NumSpatialDims number of spatial dimensions
/// @param aCellOrdinal   cell/element ordinal
/// @param aIntgPtOrdinal integration point ordinal
/// @param aIn2PKS        second piola-kirchhoff stress workset
/// @param aOut2PKS       cell second piola-kirchhoff stress
template<Plato::OrdinalType NumSpatialDims, typename ScalarType>
KOKKOS_INLINE_FUNCTION
void 
get_cell_2PKS(
  const Plato::OrdinalType                                      & aCellOrdinal,
  const Plato::OrdinalType                                      & aIntgPtOrdinal,
  const Plato::ScalarArray4DT<ScalarType>                       & aIn2PKS,
        Plato::Matrix<NumSpatialDims,NumSpatialDims,ScalarType> & aOut2PKS
)
{
  for(Plato::OrdinalType tDimI = 0; tDimI < NumSpatialDims; tDimI++){
    for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpatialDims; tDimJ++){
      aOut2PKS(tDimI,tDimJ) = aIn2PKS(aCellOrdinal,aIntgPtOrdinal,tDimI,tDimJ);
    }
  }
}

/// @brief evaluate thermo-elasto-static residual for a total lagrangian formulation:
/// \f[ 
///   R_i = \int_{\Omega_0}\delta{F}_{ij}P_{ji}\ d\Omega_0 - \int_{\Omega_0}\delta{u}_i\rho_0 b_i\ d\Omega_0 
///        -\int_{\Gamma_0}\delta{u}_i\bar{t}_i^0\ d\Gamma_0 = 0, \quad{i,j}=1,\dots,d
/// \f]
/// where \f$d\in\{1,2,3\}\f$, \f$R_i\f$ is the residual, \f$\Omega_0\f$ is the reference configuration,
/// \f$\delta{F}_{ij}\f$ is the trial deformation gradient, \f$P_{ij}\f$ is the nominal stress tensor,
/// \f$\delta{u}_i\f$ is the trial displacement field, \f$\rho_0\f$ is the mass density, \f$b_i\f$ are 
/// the body forces, $\Gamma_0$ is a boundary on body \f$\Omega_0\f$ in the reference configuration,
/// and \f$\bar{t}_i^0f\$ are the traction forces applied on boundary \f$\Gamma_0\f$. 
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types 
template<typename EvaluationType>
class ResidualThermoElastoStaticTotalLagrangian : public Plato::ResidualBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of displacement degrees of freedom per node
  static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;
  /// @brief number of displacement degrees of freedom per cell
  static constexpr auto mNumDofsPerCell = ElementType::mNumDofsPerCell;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;
  /// @brief number of integration points per cell
  static constexpr auto mNumGaussPoints = ElementType::mNumGaussPoints;
  /// @brief number of temperature degrees of freedom per node
  static constexpr auto mNumNodeStatePerNode = ElementType::mNumNodeStatePerNode;
  /// @brief local typename for base class
  using FunctionBaseType = Plato::ResidualBase;
  /// @brief contains mesh and model information
  using FunctionBaseType::mSpatialDomain;
  /// @brief output database
  using FunctionBaseType::mDataMap;
  /// @brief contains degrees of freedom names 
  using FunctionBaseType::mDofNames;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType     = typename EvaluationType::StateScalarType;
  using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
  using ResultScalarType    = typename EvaluationType::ResultScalarType;
  using ControlScalarType   = typename EvaluationType::ControlScalarType;
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  using StrainScalarType    = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief stress evaluator
  std::shared_ptr<Plato::StressEvaluator<EvaluationType>> mStressEvaluator;
  /// @brief natural boundary conditions evaluator
  std::shared_ptr<Plato::NeumannBCs<ElementType>> mNeumannBCs;
  /// @brief body loads evaluator
  std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
  /// @brief output plot table, contains requested output quantities of interests
  std::vector<std::string> mPlotTable;
  /// @brief input problem parameters
  Teuchos::ParameterList & mParamList;

public:
  /// @brief class constructor
  /// @param aSpatialDomain contains mesh and model information
  /// @param aDataMap       output database
  /// @param aParamList     input problem parameters
  ResidualThermoElastoStaticTotalLagrangian(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList
  );

  /// @brief class destructor
  ~ResidualThermoElastoStaticTotalLagrangian(){}

  /// @fn type
  /// @brief get residual type
  /// @return residual_t enum
  Plato::Elliptic::residual_t
  type() 
  const
  { return Plato::Elliptic::residual_t::NONLINEAR_THERMO_MECHANICAL; }

  /// @fn postProcess
  /// @brief post process solution database before output
  /// @param [in] aSolutions solution database
  void
  postProcess(
    const Plato::Solutions & aSolutions
  );

  /// @fn evaluate
  /// @brief evaluate internal thermo-elasto-static forces
  /// @param aWorkSets residual range and domain database
  /// @param aCycle    scalar; e.g., time step
  void
  evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) const;

  /// @fn evaluateBoundary
  /// @brief evaluate boundary forces
  /// @param aSpatialModel contains mesh and model information
  /// @param aWorkSets     reisdual range and domain database
  /// @param aCycle        scalar
  void
  evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) const;

private:
  /// @fn initialize
  /// @brief initialize member data
  /// @param aParamList 
  void 
  initialize(
    Teuchos::ParameterList & aParamList
  );
};

} // namespace Elliptic

} // namespace Plato
