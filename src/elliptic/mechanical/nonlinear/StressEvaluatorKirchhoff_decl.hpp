/*
 * StressEvaluatorKirchhoff_decl.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include <memory>

#include "MaterialModel.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/mechanical/nonlinear/StressEvaluator.hpp"

namespace Plato
{

/// @class StressEvaluatorKirchhoff
/// @brief Evaluate nominal stress for a Kirchhoff material: \n
///  \f[
///    \mathbf{P}_{ij}=S_{ik}F_{jk},\quad i,j,k=1,\dots,N_{dim}
///  \f]
/// where \f$P\f$ is the nominal stress, \f$S\f$ is the second Piola-Kirchhoff \n
/// stress tensor, and \f$F\f$ is the deformation gradient. \n
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluatorKirchhoff : public Plato::StressEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateT   = typename EvaluationType::StateScalarType;
  using ConfigT  = typename EvaluationType::ConfigScalarType;
  using ResultT  = typename EvaluationType::ResultScalarType;
  using ControlT = typename EvaluationType::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  /// @brief local typename for base class
  using BaseClassType = Plato::StressEvaluator<EvaluationType>;
  /// @brief contains mesh and model information
  using BaseClassType::mSpatialDomain;
  /// @brief output database 
  using BaseClassType::mDataMap;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName  name of material parameter list
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aSpatialDomain contains mesh and model information 
  /// @param [in] aDataMap       output database
  StressEvaluatorKirchhoff(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap
  );

  /// @brief class destructor
  ~StressEvaluatorKirchhoff(){}

  /// @fn evaluate
  /// @brief evaluate stress tensor
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  4D result workset
  /// @param [in]     aCycle   scalar 
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateT>   & aState,
      const Plato::ScalarMultiVectorT<ControlT> & aControl,
      const Plato::ScalarArray3DT<ConfigT>      & aConfig,
      const Plato::ScalarArray4DT<ResultT>      & aResult,
            Plato::Scalar                         aCycle = 0.0
  ) const;
};
    
}