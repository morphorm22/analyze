/*
 * BoundaryFluxEvaluator.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/// @class BoundaryFluxEvaluator
/// @brief base class for stress evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class BoundaryFluxEvaluator
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using ResultScalarType = typename EvaluationType::ResultScalarType;

protected:
  /// @brief name assigned to side set
  std::string mSideSetName;
  /// @brief name assigned to the material model applied on this side set
  std::string mMaterialName;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  explicit
  BoundaryFluxEvaluator(
    Teuchos::ParameterList& aNitscheParams
  )
  {
    this->initialize(aNitscheParams);
  }

  /// @brief class destructor
  virtual ~BoundaryFluxEvaluator(){}

  /// @fn evaluate
  /// @brief evaluate flux on boundary
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in]     aWorkSets     domain and range workset database
  /// @param [in,out] aResult       4D scalar container
  /// @param [in]     aCycle        scalar
  virtual 
  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const = 0;

private:
  /// @fn initialize
  /// @brief initialize member data
  /// @param [in] aParamList input problem parameters
  void 
  initialize(
    Teuchos::ParameterList & aParamList
  )
  {
    if( !aParamList.isParameter("Sides") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Sides') is not defined, ") + 
        "side set for Nitsche's method cannot be determined" )
    }
    mSideSetName = aParamList.get<std::string>("Sides");
    
    if( !aParamList.isParameter("Material Model") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Material Model') is not defined, ") + 
        "material constitutive model for Nitsche's method cannot be determined" )
    }
    mMaterialName = aParamList.get<std::string>("Material Model");
  }

};

} // namespace Plato
