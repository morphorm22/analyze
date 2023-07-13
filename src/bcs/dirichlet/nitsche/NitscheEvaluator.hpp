/*
 * NitscheEvaluator.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/// @brief parent class for nitsche integral evaluators
class NitscheEvaluator
{
protected:
  /// @brief side set name
  std::string mSideSetName;
  /// @brief name assigned to the material model applied on this side set
  std::string mMaterialName;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  NitscheEvaluator(
    Teuchos::ParameterList & aParamList
  );

  /// @brief class destructor
  virtual 
  ~NitscheEvaluator();

  /// @fn sideset
  /// @brief return side set name
  /// @return string
  std::string 
  sideset() 
  const;

  /// @fn material
  /// @brief return name assigned to material constitutive model
  /// @return string
  std::string 
  material() 
  const;

  /// @brief evaluate nitsche integral
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        scalar
  /// @param [in]     aScale        scalar 
  virtual
  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  ) = 0;

private:
  /// @fn initialize
  /// @brief initialize member data
  /// @param [in] aParamList input problem parameters
  void 
  initialize(
    Teuchos::ParameterList & aParamList
  );

};

} // namespace Plato
