/*
 * NitscheTempMisfitEvaluator_decl.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"

#include "materials/MaterialModel.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "bcs/dirichlet/nitsche/NitscheEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class NitscheTempMisfitEvaluator
/// @brief evaluate nitsche's test heat flux integral 
///
/// \f[
///   \int_{\Gamma_D}\gamma_{N}^{\theta}\delta\theta\left(\theta-\theta^{D}\right)d\Gamma
/// \f]
///
/// where \f$\gamma_{N}^{\theta}\f$ is the nitsche penalty parameter, \f$\delta\theta\f$ is the 
/// test temperature field, \f$\theta\f$ is the temperature field, \f$\theta^D\f$ is the dirichlet 
/// temperature field, and \f$\Gamma_D\f$ is the surface where dirichlet boundary conditions are enforced. 
///
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class NitscheTempMisfitEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using GradScalarType   = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief penalty for nitsche's method
  Plato::Scalar mNitschePenalty = 1.0;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:

public:
  /// @brief class constructor
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aNitscheParams input parameters for nitsche's method
  NitscheTempMisfitEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  );

  /// @fn evaluate
  /// @brief evaluate integral for all side set cells
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        scalar
  /// @param [in]     aScale        scalar
  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  );
};

} // namespace Elliptic

} // namespace Plato
