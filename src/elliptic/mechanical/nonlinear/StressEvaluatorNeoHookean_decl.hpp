/*
 * StressEvaluatorNeoHookean_decl.hpp
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

/// @class StressEvaluatorNeoHookean
/// @brief Evaluate nominal stress for a Neo-Hookean material: \n
///  \f[
///    \mathbf{P}_{ij}=S_{ik}F_{jk},\quad i,j,k=1,\dots,N_{dim}
///  \f]
/// where \f$P\f$ is the nominal stress, \f$S\f$ is the second Piola-Kirchhoff \n
/// stress tensor, and \f$F\f$ is the deformation gradient. \n
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluatorNeoHookean : public Plato::StressEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
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
  StressEvaluatorNeoHookean(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap
  );

  /// @brief class destructor
  ~StressEvaluatorNeoHookean(){}

  /// @fn evaluate
  /// @brief evaluate second piola-kirchhoff stress tensor for a neo-hookean material
  /// @param aWorkSets domain and range workset database
  /// @param aResult   4D scalar container
  /// @param aCycle    scalar
  void 
  evaluate(
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle = 0.0
  ) const;
};

} // namespace Plato
