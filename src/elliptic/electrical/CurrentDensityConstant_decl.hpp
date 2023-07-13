/*
 *  CurrentDensityConstant_decl.hpp
 *
 *  Created on: June 2, 2023
 */

#pragma once

#include "materials/MaterialModel.hpp"
#include "elliptic/electrical/CurrentDensityEvaluator.hpp"

namespace Plato
{

/// @class CurrentDensityConstant 
/// @brief evaluate constant current density model
///  \f[
///      J_i = \sigma_{ij}V_{j},\quad i,j=1,\dots,N_{dim}
///  \f]
/// where \f$J_i\f$ is the current density, \f$\sigma_{ij}\f$ is the material tensor, 
/// and \f$\V_j\f$ is the electrical potential 
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class CurrentDensityConstant : public Plato::CurrentDensityEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using GradScalarType    = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
  /// @brief local typename for base class
  using BaseClassType = Plato::CurrentDensityEvaluator<EvaluationType>;
  /// @brief contains mesh and model information
  using BaseClassType::mSpatialDomain;
  /// @brief output database 
  using BaseClassType::mDataMap;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;
  /// @brief list of requested output quantities of interest
  std::vector<std::string> mPlottable;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName  input material parameter list 
  /// @param [in] aParamList     input problem paramaters
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  CurrentDensityConstant(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap
  );

  /// @brief class destructor
  ~CurrentDensityConstant(){}

  /// @fn evaluate
  /// @brief evaluate current density
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  3D current density workset
  /// @param [in]     aCycle   scalar
  void 
  evaluate(
    const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
    const Plato::ScalarArray3DT<ResultScalarType>      & aResult,
          Plato::Scalar                                  aCycle = 0.0
  ) const;
};

} // namespace Plato
