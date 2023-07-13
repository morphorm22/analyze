/*
 * BoundaryEvaluatorTrialIsotropicElasticStress_decl.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"

#include "materials/MaterialModel.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "bcs/dirichlet/nitsche/BoundaryFluxEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{
    
/// @class BoundaryEvaluatorTrialIsotropicElasticStress
/// @brief evaluate trial stresses at the integration points using an isotropic elastic material
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class BoundaryEvaluatorTrialIsotropicElasticStress : public Plato::BoundaryFluxEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;
  /// @brief local typename for base class
  using BaseClassType = Plato::BoundaryFluxEvaluator<EvaluationType>;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;

public:
  /// @brief class constructor
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  BoundaryEvaluatorTrialIsotropicElasticStress(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );

  /// @fn evaluate
  /// @brief evaluate trial stresses for all side set cells
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in]     aWorkSets     domain and range workset database
  /// @param [in,out] aResult       4D scalar container
  /// @param [in]     aCycle        scalar
  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const;
};

} // namespace Elliptic

} // namespace Plato
