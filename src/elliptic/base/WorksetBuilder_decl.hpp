/*
 * WorksetBuilder_decl.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "SpatialModel.hpp"
#include "base/Database.hpp"
#include "base/WorksetBase.hpp"
#include "base/WorksetBuilderBase.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class WorksetBuilder
/// @brief build data worksets for elliptic partial differential equations
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class WorksetBuilder : public Plato::WorksetBuilderBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions 
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of degrees of freedom per element
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief number of nodes per element
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;  
  /// @brief interface to map vector data to worksets
  const Plato::WorksetBase<ElementType> & mWorksetFuncs;

public:
  /// @brief class constructor
  /// @param aWorksetFuncs interface to map vector data to worksets
  WorksetBuilder(
    const Plato::WorksetBase<ElementType> & aWorksetFuncs
  );

  /// @brief class destructor
  ~WorksetBuilder(){}

  /// @fn build
  /// @brief build worksets for elliptic problems
  /// @param [in]     aDomain   contains mesh and model information
  /// @param [in]     aDatabase function domain and range database
  /// @param [in,out] aWorkSets workset database
  void build(
    const Plato::SpatialDomain & aDomain,
    const Plato::Database      & aDatabase,
          Plato::WorkSets      & aWorkSets
  ) const;

  /// @fn build
  /// @brief build worksets for elliptic problems
  /// @param [in]     aNumCells local number of cells/elements
  /// @param [in]     aDatabase function domain and range database
  /// @param [in,out] aWorkSets workset database
  void build(
    const Plato::OrdinalType & aNumCells,
    const Plato::Database    & aDatabase,
          Plato::WorkSets    & aWorkSets
  ) const;

}; // class WorksetBuilder

} // namespace Elliptic

} // namespace Plato