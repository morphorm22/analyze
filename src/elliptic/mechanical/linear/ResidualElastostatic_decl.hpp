#pragma once

#include <memory>

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "CellForcing.hpp"
#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"

#include "base/ResidualBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Elastostatic vector function interface
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * \tparam IndicatorFunctionType penalty function used for density-based methods
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ResidualElastostatic : public Plato::ResidualBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of stress-strain terms
  static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;
  /// @brief number of integration points
  static constexpr auto mNumGaussPoints  = ElementType::mNumGaussPoints;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  
  using FunctionBaseType = Plato::ResidualBase;
  using FunctionBaseType::mSpatialDomain;
  using FunctionBaseType::mDataMap;
  using FunctionBaseType::mDofNames;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

  IndicatorFunctionType mIndicatorFunction;
  Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;
  Plato::CellForcing<ElementType> mCellForcing;

  std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
  std::shared_ptr<Plato::NaturalBCs<ElementType>> mBoundaryLoads;

  Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

  std::vector<std::string> mPlotTable;

public:
  /******************************************************************************//**
   * \brief Constructor
   * \param [in] aSpatialDomain Plato Analyze spatial domain
   * \param [in] aDataMap Plato Analyze database
   * \param [in] aProblemParams input parameters for overall problem
   * \param [in] aPenaltyParams input parameters for penalty function
  **********************************************************************************/
  ResidualElastostatic(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          Teuchos::ParameterList & aPenaltyParams
  );

  /// @fn type
  /// @brief get residual type
  /// @return residual_t enum
  Plato::Elliptic::residual_t
  type() 
  const
  { return Plato::Elliptic::residual_t::LINEAR_MECHANICAL; }

  /****************************************************************************//**
  * \brief Pure virtual function to get output solution data
  * \param [in] state solution database
  * \return output state solution database
  ********************************************************************************/
  Plato::Solutions 
  getSolutionStateOutputData(
    const Plato::Solutions &aSolutions
  ) const;

  /// @fn evaluate
  /// @brief evaluate elastostatics residual, internal forces only
  /// @param [in,out] aWorkSets domain and range workset database
  /// @param [in]     aCycle    scalar
  void
  evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) const;

  /// @fn evaluateBoundary
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

  /// @fn outputVonMises
  /// @brief compute Von Mises stresses and save in output database
  /// @param [in] aCauchyStress  cauchy stress
  /// @param [in] aSpatialDomain contains mesh and model information
  void
  outputVonMises(
    const Plato::ScalarMultiVectorT<ResultScalarType> & aCauchyStress,
    const Plato::SpatialDomain                        & aSpatialDomain
  ) const;

};
// class ResidualElastostatic

} // namespace Elliptic

} // namespace Plato
